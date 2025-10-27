#!/usr/bin/env python3
import traci
import random
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ================================
# CONFIGURAÇÕES DO AMBIENTE
# ================================
SUMO_CFG_FILE = "teste4_sumo.sumocfg"
TRAFFIC_LIGHT_IDS = ["B2", "C2", "D2"]   # IDs dos semáforos no SUMO

ACTIONS = ["green_h", "green_v"]  # Ações possíveis
EPSILON = 0.1   # Taxa de exploração
ALPHA = 0.1     # Taxa de aprendizado
GAMMA = 0.9     # Fator de desconto

# ======================================
# DEFINIÇÃO DAS VARIÁVEIS FUZZY
# ======================================
# Entradas fuzzy: tamanho da fila (queue) e tempo de espera (wait)
queue = ctrl.Antecedent(np.arange(0, 31, 1), 'queue')
wait = ctrl.Antecedent(np.arange(0, 101, 1), 'wait')

# Saída fuzzy: recompensa
reward_out = ctrl.Consequent(np.arange(-100, 1, 1), 'reward')

# Funções de pertinência para filas
queue['low'] = fuzz.trimf(queue.universe, [0, 0, 10])
queue['medium'] = fuzz.trimf(queue.universe, [5, 15, 25])
queue['high'] = fuzz.trimf(queue.universe, [20, 30, 30])

# Funções de pertinência para tempos de espera
wait['low'] = fuzz.trimf(wait.universe, [0, 0, 30])
wait['medium'] = fuzz.trimf(wait.universe, [20, 50, 80])
wait['high'] = fuzz.trimf(wait.universe, [60, 100, 100])

# Funções de pertinência para a recompensa
reward_out['excellent'] = fuzz.trimf(reward_out.universe, [-5, 0, 0])
reward_out['good'] = fuzz.trimf(reward_out.universe, [-20, -10, 0])
reward_out['fair'] = fuzz.trimf(reward_out.universe, [-50, -30, -10])
reward_out['poor'] = fuzz.trimf(reward_out.universe, [-80, -60, -40])
reward_out['very_poor'] = fuzz.trimf(reward_out.universe, [-100, -90, -70])

# ======================================
# REGRAS FUZZY
# ======================================
rule1 = ctrl.Rule(queue['low'] & wait['low'], reward_out['excellent'])
rule2 = ctrl.Rule(queue['low'] & wait['medium'], reward_out['good'])
rule3 = ctrl.Rule(queue['low'] & wait['high'], reward_out['fair'])
rule4 = ctrl.Rule(queue['medium'] & wait['low'], reward_out['good'])
rule5 = ctrl.Rule(queue['medium'] & wait['medium'], reward_out['fair'])
rule6 = ctrl.Rule(queue['medium'] & wait['high'], reward_out['poor'])
rule7 = ctrl.Rule(queue['high'] & wait['low'], reward_out['fair'])
rule8 = ctrl.Rule(queue['high'] & wait['medium'], reward_out['poor'])
rule9 = ctrl.Rule(queue['high'] & wait['high'], reward_out['very_poor'])

# Controlador fuzzy
reward_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
reward_simulator = ctrl.ControlSystemSimulation(reward_ctrl)

# ================================
# FUNÇÕES AUXILIARES
# ================================
def get_queue_and_wait(tl_id):
    """
    Retorna o número de veículos na fila (horizontal e vertical)
    e o tempo de espera acumulado (horizontal e vertical).
    """
    q_h, q_v, w_h, w_v = 0, 0, 0, 0
    for edge in traci.trafficlight.getControlledLanes(tl_id):
        for veh in traci.lane.getLastStepVehicleIDs(edge):
            if "h" in veh:
                q_h += 1
                w_h += traci.vehicle.getWaitingTime(veh)
            elif "v" in veh:
                q_v += 1
                w_v += traci.vehicle.getWaitingTime(veh)
    return q_h, q_v, w_h, w_v


def get_state(tl_id):
    """
    Retorna o estado discretizado para evitar explosão da Q-table.
    """
    qh, qv, wh, wv = get_queue_and_wait(tl_id)
    return (qh // 3, qv // 3, wh // 10, wv // 10)


def compute_reward(tl_id):
    """
    Calcula a recompensa fuzzy a partir das filas e tempos de espera.
    """
    qh, qv, wh, wv = get_queue_and_wait(tl_id)
    reward_simulator.input['queue'] = qh + qv
    reward_simulator.input['wait'] = wh + wv
    reward_simulator.compute()
    return reward_simulator.output['reward']


def get_q(Q, state, action):
    """Retorna o valor Q(s,a) armazenado ou 0.0 se não existir."""
    return Q.get((state, action), 0.0)


def set_q(Q, state, action, value):
    """Define o valor Q(s,a)."""
    Q[(state, action)] = value


def epsilon_greedy(Q, state):
    """
    Seleciona uma ação com base na política epsilon-greedy:
    - Exploração (aleatória) com probabilidade EPSILON
    - Exploitação (melhor ação conhecida) com probabilidade 1-EPSILON
    """
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)
    qs = [get_q(Q, state, a) for a in ACTIONS]
    return ACTIONS[np.argmax(qs)]


def q_learning(Q, state, action, reward, next_state):
    """
    Atualiza a Q-table usando a fórmula do Q-Learning.
    """
    q_sa = get_q(Q, state, action)
    max_next_q = max([get_q(Q, next_state, a) for a in ACTIONS], default=0.0)
    new_q = q_sa + ALPHA * (reward + GAMMA * max_next_q - q_sa)
    set_q(Q, state, action, new_q)


def set_phase(tl_id, action):
    """
    Define a fase do semáforo de acordo com a ação escolhida.
    """
    if action == "green_h":
        traci.trafficlight.setPhase(tl_id, 0)  # Verde horizontal
    elif action == "green_v":
        traci.trafficlight.setPhase(tl_id, 2)  # Verde vertical

# ================================
# TREINAMENTO
# ================================
def run_episode(Q):
    """
    Executa um episódio de treinamento no SUMO.
    """
    traci.start(["sumo", "-c", SUMO_CFG_FILE, "--start", "--no-step-log", "true"])
    step = 0
    while step < 1000:  # Número de steps do episódio
        for tl in TRAFFIC_LIGHT_IDS:
            state = get_state(tl)
            action = epsilon_greedy(Q, state)
            set_phase(tl, action)

            traci.simulationStep()

            reward = compute_reward(tl)
            next_state = get_state(tl)
            q_learning(Q, state, action, reward, next_state)

        step += 1
    traci.close()


def main():
    """
    Função principal: executa múltiplos episódios de treinamento.
    """
    Q = {}
    for ep in range(100):   # treina em 3 episódios (ajuste conforme necessário)
        print(f"Treinando episódio {ep+1}...")
        run_episode(Q)
    print("Treinamento concluído.")
    return Q


if __name__ == "__main__":
    main()
