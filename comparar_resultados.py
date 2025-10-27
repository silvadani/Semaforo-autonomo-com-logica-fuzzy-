import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Arquivos de resultado
fixed_files = {
    'carros_parados': "resultado_tempo_fixo.csv",
    'total_paradas': "paradas_tempo_fixo.csv",
    'tempo_espera': "espera_tempo_fixo.csv",
    'velocidade_media': "velocidade_tempo_fixo.csv",
    'tempo_espera_emergency': "emergency_tempo_fixo.csv",
    'tempo_espera_authority': "authority_tempo_fixo.csv",
    'carros_parados_prioritarios': "carros_parados_prioritarios_tempo_fixo.csv",
    'total_paradas_prioritarios': "paradas_prioritarios_tempo_fixo.csv",
    'tempo_espera_prioritarios': "espera_prioritarios_tempo_fixo.csv",
    'velocidade_media_prioritarios': "velocidade_prioritarios_tempo_fixo.csv"
}
rl_files = {
    'carros_parados': "resultado_qlearning.csv",
    'total_paradas': "paradas_qlearning.csv",
    'tempo_espera': "espera_qlearning.csv",
    'velocidade_media': "velocidade_qlearning.csv",
    'tempo_espera_emergency': "emergency_qlearning.csv",
    'tempo_espera_authority': "authority_qlearning.csv",
    'carros_parados_prioritarios': "carros_parados_prioritarios_qlearning.csv",
    'total_paradas_prioritarios': "paradas_prioritarios_qlearning.csv",
    'tempo_espera_prioritarios': "espera_prioritarios_qlearning.csv",
    'velocidade_media_prioritarios': "velocidade_prioritarios_qlearning.csv"
}

# Diretório de saída
output_dir = "relatorio"
os.makedirs(output_dir, exist_ok=True)

# Leitura dos dados
dfs_fixed = {}
dfs_rl = {}
for key, file in fixed_files.items():
    try:
        dfs_fixed[key] = pd.read_csv(file)
    except FileNotFoundError:
        print(f"Arquivo {file} não encontrado.")
        dfs_fixed[key] = pd.DataFrame()

for key, file in rl_files.items():
    try:
        dfs_rl[key] = pd.read_csv(file)
    except FileNotFoundError:
        print(f"Arquivo {file} não encontrado.")
        dfs_rl[key] = pd.DataFrame()

# Padding dos dados para igualar os tempos
for metric in ['carros_parados', 'total_paradas', 'tempo_espera', 'velocidade_media', 'carros_parados_prioritarios', 'total_paradas_prioritarios', 'tempo_espera_prioritarios', 'velocidade_media_prioritarios']:
    df_fixed = dfs_fixed[metric]
    df_rl = dfs_rl[metric]
    if not df_fixed.empty and not df_rl.empty:
        max_time = int(max(df_fixed['tempo'].max(), df_rl['tempo'].max()))
        min_fixed = df_fixed[metric].min()
        min_rl = df_rl[metric].min()
        # Pad fixed if shorter
        if df_fixed['tempo'].max() < max_time:
            last_time = int(df_fixed['tempo'].max())
            for t in range(last_time + 1, max_time + 1):
                new_row = {'tempo': t, metric: min_fixed}
                df_fixed = pd.concat([df_fixed, pd.DataFrame([new_row])], ignore_index=True)
        # Pad rl if shorter
        if df_rl['tempo'].max() < max_time:
            last_time = int(df_rl['tempo'].max())
            for t in range(last_time + 1, max_time + 1):
                new_row = {'tempo': t, metric: min_rl}
                df_rl = pd.concat([df_rl, pd.DataFrame([new_row])], ignore_index=True)
        dfs_fixed[metric] = df_fixed
        dfs_rl[metric] = df_rl

def get_column(metric):
    column_map = {
        'tempo_espera_emergency': 'media_espera_emergency',
        'tempo_espera_authority': 'media_espera_authority',
        'carros_parados': 'carros_parados',
        'total_paradas': 'total_paradas',
        'tempo_espera': 'tempo_espera',
        'velocidade_media': 'velocidade_media',
        'carros_parados_prioritarios': 'carros_parados_prioritarios',
        'total_paradas_prioritarios': 'total_paradas_prioritarios',
        'tempo_espera_prioritarios': 'tempo_espera_prioritarios',
        'velocidade_media_prioritarios': 'velocidade_media_prioritarios'
    }
    return column_map.get(metric, metric)

# Cálculo de métricas agregadas
def compute_metrics(df, metric):
    if df.empty:
        return {'media': 0, 'desvio_padrao': 0, 'maximo': 0, 'minimo': 0}
    column = get_column(metric)
    return {
        'media': df[column].mean(),
        'desvio_padrao': df[column].std(),
        'maximo': df[column].max(),
        'minimo': df[column].min()
    }

metrics_fixed = {key: compute_metrics(df, key) for key, df in dfs_fixed.items()}
metrics_rl = {key: compute_metrics(df, key) for key, df in dfs_rl.items()}

# Plot comparativo para cada métrica
metric_labels = {
    'carros_parados': 'Número de Carros Parados',
    'total_paradas': 'Total de Paradas (Estimativa)',
    'tempo_espera': 'Tempo Médio de Espera (s)',
    'velocidade_media': 'Velocidade Média (m/s)',
    'tempo_espera_emergency': 'Tempo de Espera Médio - Emergência (s)',
    'tempo_espera_authority': 'Tempo de Espera Médio - Autoridade (s)',
    'carros_parados_prioritarios': 'Número de Carros Parados - Prioritários',
    'total_paradas_prioritarios': 'Total de Paradas - Prioritários (Estimativa)',
    'tempo_espera_prioritarios': 'Tempo Médio de Espera - Prioritários (s)',
    'velocidade_media_prioritarios': 'Velocidade Média - Prioritários (m/s)'
}
for metric in ['carros_parados', 'total_paradas', 'tempo_espera', 'velocidade_media', 'tempo_espera_emergency', 'tempo_espera_authority', 'carros_parados_prioritarios', 'total_paradas_prioritarios', 'tempo_espera_prioritarios', 'velocidade_media_prioritarios']:
    plt.figure(figsize=(12, 6))
    if not dfs_fixed[metric].empty:
        plt.plot(dfs_fixed[metric]['tempo'], dfs_fixed[metric][get_column(metric)], label='Tempo Fixo (Controle Tradicional)', color='blue', linewidth=2)
    if not dfs_rl[metric].empty:
        plt.plot(dfs_rl[metric]['tempo'], dfs_rl[metric][get_column(metric)], label='Q-Learning (Aprendizado por Reforço)', color='red', linewidth=2)
    plt.title(f'Comparação de {metric_labels[metric]} ao Longo do Tempo', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
    plt.ylabel(metric_labels[metric], fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'comparacao_{metric}.png')
# Gráficos individuais para prioritários
# Emergency - Tempo Fixo
plt.figure(figsize=(10, 6))
plt.plot(dfs_fixed['tempo_espera_emergency']['tempo'], dfs_fixed['tempo_espera_emergency']['media_espera_emergency'], label='Tempo Fixo', color='blue')
plt.title('Tempo de Espera - Emergência - Tempo Fixo', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Tempo de Espera (s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_tempo_espera_emergency_tempo_fixo.png'), dpi=300)
plt.close()

# Emergency - Q-Learning
plt.figure(figsize=(10, 6))
plt.plot(dfs_rl['tempo_espera_emergency']['tempo'], dfs_rl['tempo_espera_emergency']['media_espera_emergency'], label='Q-Learning', color='red')
plt.title('Tempo de Espera - Emergência - Q-Learning', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Tempo de Espera (s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_tempo_espera_emergency_qlearning.png'), dpi=300)
plt.close()

# Authority - Tempo Fixo
plt.figure(figsize=(10, 6))
plt.plot(dfs_fixed['tempo_espera_authority']['tempo'], dfs_fixed['tempo_espera_authority']['media_espera_authority'], label='Tempo Fixo', color='blue')
plt.title('Tempo de Espera - Autoridade - Tempo Fixo', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Tempo de Espera (s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_tempo_espera_authority_tempo_fixo.png'), dpi=300)
plt.close()

# Authority - Q-Learning
plt.figure(figsize=(10, 6))
plt.plot(dfs_rl['tempo_espera_authority']['tempo'], dfs_rl['tempo_espera_authority']['media_espera_authority'], label='Q-Learning', color='red')
plt.title('Tempo de Espera - Autoridade - Q-Learning', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Tempo de Espera (s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_tempo_espera_authority_qlearning.png'), dpi=300)
plt.close()
labels = ['Carros Parados', 'Total de Paradas', 'Tempo de Espera', 'Velocidade Média']
fixed_means = [metrics_fixed[m]['media'] for m in ['carros_parados', 'total_paradas', 'tempo_espera', 'velocidade_media']]
rl_means = [metrics_rl[m]['media'] for m in ['carros_parados', 'total_paradas', 'tempo_espera', 'velocidade_media']]

x = range(len(labels))
width = 0.35
plt.figure(figsize=(12, 6))
bars1 = plt.bar([i - width/2 for i in x], fixed_means, width, label='Tempo Fixo', color='skyblue', edgecolor='black')
bars2 = plt.bar([i + width/2 for i in x], rl_means, width, label='Q-Learning', color='salmon', edgecolor='black')

plt.xlabel('Métricas de Desempenho', fontsize=14)
plt.ylabel('Valor Médio', fontsize=14)
plt.title('Comparação de Médias: Tempo Fixo vs Q-Learning', fontsize=16, fontweight='bold')
plt.xticks(x, labels, fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
bar_plot_path = os.path.join(output_dir, 'comparacao_medias.png')
plt.savefig(bar_plot_path, dpi=300)
plt.close()

# Gráfico de barras para Prioritários
labels_prioritarios = ['Tempo Espera Emergência', 'Tempo Espera Autoridade', 'Carros Parados Prioritários', 'Total Paradas Prioritários', 'Tempo Espera Prioritários', 'Velocidade Média Prioritários']
fixed_means_prioritarios = [metrics_fixed[m]['media'] for m in ['tempo_espera_emergency', 'tempo_espera_authority', 'carros_parados_prioritarios', 'total_paradas_prioritarios', 'tempo_espera_prioritarios', 'velocidade_media_prioritarios']]
rl_means_prioritarios = [metrics_rl[m]['media'] for m in ['tempo_espera_emergency', 'tempo_espera_authority', 'carros_parados_prioritarios', 'total_paradas_prioritarios', 'tempo_espera_prioritarios', 'velocidade_media_prioritarios']]

x_prioritarios = range(len(labels_prioritarios))
width = 0.35
plt.figure(figsize=(14, 8))
bars1_prioritarios = plt.bar([i - width/2 for i in x_prioritarios], fixed_means_prioritarios, width, label='Tempo Fixo', color='skyblue', edgecolor='black')
bars2_prioritarios = plt.bar([i + width/2 for i in x_prioritarios], rl_means_prioritarios, width, label='Q-Learning', color='salmon', edgecolor='black')

plt.xlabel('Métricas de Desempenho - Prioritários', fontsize=14)
plt.ylabel('Valor Médio', fontsize=14)
plt.title('Comparação de Médias - Prioritários: Tempo Fixo vs Q-Learning', fontsize=16, fontweight='bold')
plt.xticks(x_prioritarios, labels_prioritarios, fontsize=10, rotation=45, ha='right')
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar in bars1_prioritarios:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)
for bar in bars2_prioritarios:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
bar_plot_prioritarios_path = os.path.join(output_dir, 'comparacao_medias_prioritarios.png')
plt.savefig(bar_plot_prioritarios_path, dpi=300)
plt.close()

# Gráfico de barras Geral vs Prioritários
labels_geral_vs_prior = ['Carros Parados Geral', 'Carros Parados Prioritários', 'Total Paradas Geral', 'Total Paradas Prioritários', 'Tempo Espera Geral', 'Tempo Espera Prioritários', 'Velocidade Geral', 'Velocidade Prioritários']
fixed_means_geral_vs_prior = [
    metrics_fixed['carros_parados']['media'], metrics_fixed['carros_parados_prioritarios']['media'],
    metrics_fixed['total_paradas']['media'], metrics_fixed['total_paradas_prioritarios']['media'],
    metrics_fixed['tempo_espera']['media'], metrics_fixed['tempo_espera_prioritarios']['media'],
    metrics_fixed['velocidade_media']['media'], metrics_fixed['velocidade_media_prioritarios']['media']
]
rl_means_geral_vs_prior = [
    metrics_rl['carros_parados']['media'], metrics_rl['carros_parados_prioritarios']['media'],
    metrics_rl['total_paradas']['media'], metrics_rl['total_paradas_prioritarios']['media'],
    metrics_rl['tempo_espera']['media'], metrics_rl['tempo_espera_prioritarios']['media'],
    metrics_rl['velocidade_media']['media'], metrics_rl['velocidade_media_prioritarios']['media']
]

x_geral_vs_prior = range(len(labels_geral_vs_prior))
width = 0.35
plt.figure(figsize=(16, 8))
bars1_geral_vs_prior = plt.bar([i - width/2 for i in x_geral_vs_prior], fixed_means_geral_vs_prior, width, label='Tempo Fixo', color='skyblue', edgecolor='black')
bars2_geral_vs_prior = plt.bar([i + width/2 for i in x_geral_vs_prior], rl_means_geral_vs_prior, width, label='Q-Learning', color='salmon', edgecolor='black')

plt.xlabel('Métricas: Geral vs Prioritários', fontsize=14)
plt.ylabel('Valor Médio', fontsize=14)
plt.title('Comparação de Médias: Geral vs Prioritários - Tempo Fixo vs Q-Learning', fontsize=16, fontweight='bold')
plt.xticks(x_geral_vs_prior, labels_geral_vs_prior, fontsize=10, rotation=45, ha='right')
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar in bars1_geral_vs_prior:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)
for bar in bars2_geral_vs_prior:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
bar_plot_geral_vs_prior_path = os.path.join(output_dir, 'comparacao_geral_vs_prioritarios.png')
plt.savefig(bar_plot_geral_vs_prior_path, dpi=300)
plt.close()

# Gráficos comparativos para métricas prioritárias
for metric in ['carros_parados_prioritarios', 'total_paradas_prioritarios', 'tempo_espera_prioritarios', 'velocidade_media_prioritarios']:
    plt.figure(figsize=(12, 6))
    plt.plot(dfs_fixed[metric]['tempo'], dfs_fixed[metric][metric], label='Tempo Fixo', color='blue')
    plt.plot(dfs_rl[metric]['tempo'], dfs_rl[metric][metric], label='Q-Learning', color='red')
    plt.title(f'Comparação de {metric_labels[metric]} - Tempo Fixo vs Q-Learning', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
    plt.ylabel(metric_labels[metric], fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparacao_{metric}.png'), dpi=300)
    plt.close()

# Geração de relatório em HTML
def gerar_html(metrics_fixed, metrics_rl, output_file):
    html = """
    <html>
      <head><title>Relatório de Comparação</title></head>
      <body>
        <h1>Relatório de Comparação: Tempo Fixo vs Q-Learning</h1>
        <p><strong>Interpretação:</strong> Este relatório compara o controle de semáforos tradicional (tempo fixo) com o aprendizado por reforço (Q-Learning). Valores menores em "Carros Parados", "Total de Paradas" e "Tempo de Espera" indicam melhor desempenho. Valores maiores em "Velocidade Média" são melhores.</p>
    """
    for metric in ['carros_parados', 'total_paradas', 'tempo_espera', 'velocidade_media']:
        html += """
        <h2>Métricas para {}</h2>
        <table border='1' cellpadding='5'>
          <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
          <tr><td>Tempo Fixo</td><td>{:.2f}</td><td>{:.2f}</td><td>{}</td><td>{}</td></tr>
          <tr><td>Q-Learning</td><td>{:.2f}</td><td>{:.2f}</td><td>{}</td><td>{}</td></tr>
        </table>
        <p><em>Gráfico mostra a evolução ao longo do tempo. Linha azul: Tempo Fixo. Linha vermelha: Q-Learning.</em></p>
        <img src='comparacao_{}.png' alt='Comparação de {}'>
        """.format(
            metric_labels[metric],
            metrics_fixed[metric]['media'], metrics_fixed[metric]['desvio_padrao'], metrics_fixed[metric]['maximo'], metrics_fixed[metric]['minimo'],
            metrics_rl[metric]['media'], metrics_rl[metric]['desvio_padrao'], metrics_rl[metric]['maximo'], metrics_rl[metric]['minimo'],
            metric, metric_labels[metric]
        )
    html += """
        <h2>Comparação de Médias</h2>
        <p><em>Gráfico de barras mostra as médias gerais. Azul: Tempo Fixo. Vermelho: Q-Learning. Valores nas barras indicam as médias exatas.</em></p>
        <img src='comparacao_medias.png' alt='Comparação de Médias'>
      </body>
    </html>
    """
    with open(output_file, 'w') as f:
        f.write(html)
# Emergency - Tempo Fixo
plt.figure(figsize=(10, 6))
plt.plot(dfs_fixed['tempo_espera_emergency']['tempo'], dfs_fixed['tempo_espera_emergency']['media_espera_emergency'], label='Tempo Fixo', color='blue')
plt.title('Tempo de Espera - Emergência - Tempo Fixo', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Tempo de Espera (s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_tempo_espera_emergency_tempo_fixo.png'), dpi=300)
plt.close()

# Emergency - Q-Learning
plt.figure(figsize=(10, 6))
plt.plot(dfs_rl['tempo_espera_emergency']['tempo'], dfs_rl['tempo_espera_emergency']['media_espera_emergency'], label='Q-Learning', color='red')
plt.title('Tempo de Espera - Emergência - Q-Learning', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Tempo de Espera (s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_tempo_espera_emergency_qlearning.png'), dpi=300)
plt.close()

# Authority - Tempo Fixo
plt.figure(figsize=(10, 6))
plt.plot(dfs_fixed['tempo_espera_authority']['tempo'], dfs_fixed['tempo_espera_authority']['media_espera_authority'], label='Tempo Fixo', color='blue')
plt.title('Tempo de Espera - Autoridade - Tempo Fixo', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Tempo de Espera (s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_tempo_espera_authority_tempo_fixo.png'), dpi=300)
plt.close()

# Authority - Q-Learning
plt.figure(figsize=(10, 6))
plt.plot(dfs_rl['tempo_espera_authority']['tempo'], dfs_rl['tempo_espera_authority']['media_espera_authority'], label='Q-Learning', color='red')
plt.title('Tempo de Espera - Autoridade - Q-Learning', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Tempo de Espera (s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_tempo_espera_authority_qlearning.png'), dpi=300)
plt.close()

# Carros Parados Prioritários - Tempo Fixo
plt.figure(figsize=(10, 6))
plt.plot(dfs_fixed['carros_parados_prioritarios']['tempo'], dfs_fixed['carros_parados_prioritarios']['carros_parados_prioritarios'], label='Tempo Fixo', color='blue')
plt.title('Carros Parados Prioritários - Tempo Fixo', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Número de Carros Parados', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_carros_parados_prioritarios_tempo_fixo.png'), dpi=300)
plt.close()

# Carros Parados Prioritários - Q-Learning
plt.figure(figsize=(10, 6))
plt.plot(dfs_rl['carros_parados_prioritarios']['tempo'], dfs_rl['carros_parados_prioritarios']['carros_parados_prioritarios'], label='Q-Learning', color='red')
plt.title('Carros Parados Prioritários - Q-Learning', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Número de Carros Parados', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_carros_parados_prioritarios_qlearning.png'), dpi=300)
plt.close()

# Total Paradas Prioritários - Tempo Fixo
plt.figure(figsize=(10, 6))
plt.plot(dfs_fixed['total_paradas_prioritarios']['tempo'], dfs_fixed['total_paradas_prioritarios']['total_paradas_prioritarios'], label='Tempo Fixo', color='blue')
plt.title('Total Paradas Prioritários - Tempo Fixo', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Total de Paradas', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_total_paradas_prioritarios_tempo_fixo.png'), dpi=300)
plt.close()

# Total Paradas Prioritários - Q-Learning
plt.figure(figsize=(10, 6))
plt.plot(dfs_rl['total_paradas_prioritarios']['tempo'], dfs_rl['total_paradas_prioritarios']['total_paradas_prioritarios'], label='Q-Learning', color='red')
plt.title('Total Paradas Prioritários - Q-Learning', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Total de Paradas', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_total_paradas_prioritarios_qlearning.png'), dpi=300)
plt.close()

# Tempo Espera Prioritários - Tempo Fixo
plt.figure(figsize=(10, 6))
plt.plot(dfs_fixed['tempo_espera_prioritarios']['tempo'], dfs_fixed['tempo_espera_prioritarios']['tempo_espera_prioritarios'], label='Tempo Fixo', color='blue')
plt.title('Tempo Espera Prioritários - Tempo Fixo', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Tempo de Espera (s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_tempo_espera_prioritarios_tempo_fixo.png'), dpi=300)
plt.close()

# Tempo Espera Prioritários - Q-Learning
plt.figure(figsize=(10, 6))
plt.plot(dfs_rl['tempo_espera_prioritarios']['tempo'], dfs_rl['tempo_espera_prioritarios']['tempo_espera_prioritarios'], label='Q-Learning', color='red')
plt.title('Tempo Espera Prioritários - Q-Learning', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Tempo de Espera (s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_tempo_espera_prioritarios_qlearning.png'), dpi=300)
plt.close()

# Velocidade Media Prioritários - Tempo Fixo
plt.figure(figsize=(10, 6))
plt.plot(dfs_fixed['velocidade_media_prioritarios']['tempo'], dfs_fixed['velocidade_media_prioritarios']['velocidade_media_prioritarios'], label='Tempo Fixo', color='blue')
plt.title('Velocidade Media Prioritários - Tempo Fixo', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Velocidade (m/s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_velocidade_media_prioritarios_tempo_fixo.png'), dpi=300)
plt.close()

# Velocidade Media Prioritários - Q-Learning
plt.figure(figsize=(10, 6))
plt.plot(dfs_rl['velocidade_media_prioritarios']['tempo'], dfs_rl['velocidade_media_prioritarios']['velocidade_media_prioritarios'], label='Q-Learning', color='red')
plt.title('Velocidade Media Prioritários - Q-Learning', fontsize=16, fontweight='bold')
plt.xlabel('Tempo de Simulação (segundos)', fontsize=14)
plt.ylabel('Velocidade (m/s)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_velocidade_media_prioritarios_qlearning.png'), dpi=300)
plt.close()

# Gerar relatório separado para Tempo Fixo - Prioritários
html_fixed = """
<html>
<head>
    <title>Relatório de Tratamento de Veículos Prioritários - Tempo Fixo</title>
</head>
<body>
    <h1>Relatório de Tratamento de Veículos Prioritários - Tempo Fixo</h1>
    <h2>Métricas para Tempo de Espera - Emergência</h2>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para tempo de espera de emergências.</em></p>
    <img src='comparacao_tempo_espera_emergency_tempo_fixo.png' alt='Tempo de Espera - Emergência - Tempo Fixo'>
    
    <h2>Métricas para Tempo de Espera - Autoridade</h2>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para tempo de espera de autoridades.</em></p>
    <img src='comparacao_tempo_espera_authority_tempo_fixo.png' alt='Tempo de Espera - Autoridade - Tempo Fixo'>
    
    <h2>Métricas Gerais para Veículos Prioritários</h2>
    <h3>Carros Parados Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para carros parados prioritários.</em></p>
    <img src='comparacao_carros_parados_prioritarios_tempo_fixo.png' alt='Carros Parados Prioritários - Tempo Fixo'>
    
    <h3>Total Paradas Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para total paradas prioritários.</em></p>
    <img src='comparacao_total_paradas_prioritarios_tempo_fixo.png' alt='Total Paradas Prioritários - Tempo Fixo'>
    
    <h3>Tempo Espera Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para tempo espera prioritários.</em></p>
    <img src='comparacao_tempo_espera_prioritarios_tempo_fixo.png' alt='Tempo Espera Prioritários - Tempo Fixo'>
    
    <h3>Velocidade Média Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para velocidade média prioritários.</em></p>
    <img src='comparacao_velocidade_media_prioritarios_tempo_fixo.png' alt='Velocidade Média Prioritários - Tempo Fixo'>
</body>
</html>
""".format(
    f"{metrics_fixed['tempo_espera_emergency']['media']:.2f}",
    f"{metrics_fixed['tempo_espera_emergency']['desvio_padrao']:.2f}",
    f"{metrics_fixed['tempo_espera_emergency']['maximo']:.2f}",
    f"{metrics_fixed['tempo_espera_emergency']['minimo']:.2f}",
    f"{metrics_fixed['tempo_espera_authority']['media']:.2f}",
    f"{metrics_fixed['tempo_espera_authority']['desvio_padrao']:.2f}",
    f"{metrics_fixed['tempo_espera_authority']['maximo']:.2f}",
    f"{metrics_fixed['tempo_espera_authority']['minimo']:.2f}",
    f"{metrics_fixed['carros_parados_prioritarios']['media']:.2f}",
    f"{metrics_fixed['carros_parados_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_fixed['carros_parados_prioritarios']['maximo']:.2f}",
    f"{metrics_fixed['carros_parados_prioritarios']['minimo']:.2f}",
    f"{metrics_fixed['total_paradas_prioritarios']['media']:.2f}",
    f"{metrics_fixed['total_paradas_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_fixed['total_paradas_prioritarios']['maximo']:.2f}",
    f"{metrics_fixed['total_paradas_prioritarios']['minimo']:.2f}",
    f"{metrics_fixed['tempo_espera_prioritarios']['media']:.2f}",
    f"{metrics_fixed['tempo_espera_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_fixed['tempo_espera_prioritarios']['maximo']:.2f}",
    f"{metrics_fixed['tempo_espera_prioritarios']['minimo']:.2f}",
    f"{metrics_fixed['velocidade_media_prioritarios']['media']:.2f}",
    f"{metrics_fixed['velocidade_media_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_fixed['velocidade_media_prioritarios']['maximo']:.2f}",
    f"{metrics_fixed['velocidade_media_prioritarios']['minimo']:.2f}"
)

with open("relatorio/relatorio_prioritarios_tempo_fixo.html", "w") as f:
    f.write(html_fixed)

# Gerar relatório separado para Q-Learning - Prioritários
html_rl = """
<html>
<head>
    <title>Relatório de Tratamento de Veículos Prioritários - Q-Learning</title>
</head>
<body>
    <h1>Relatório de Tratamento de Veículos Prioritários - Q-Learning</h1>
    <h2>Métricas para Tempo de Espera - Emergência</h2>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para tempo de espera de emergências.</em></p>
    <img src='comparacao_tempo_espera_emergency_qlearning.png' alt='Tempo de Espera - Emergência - Q-Learning'>
    
    <h2>Métricas para Tempo de Espera - Autoridade</h2>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para tempo de espera de autoridades.</em></p>
    <img src='comparacao_tempo_espera_authority_qlearning.png' alt='Tempo de Espera - Autoridade - Q-Learning'>
    
    <h2>Métricas Gerais para Veículos Prioritários</h2>
    <h3>Carros Parados Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para carros parados prioritários.</em></p>
    <img src='comparacao_carros_parados_prioritarios_qlearning.png' alt='Carros Parados Prioritários - Q-Learning'>
    
    <h3>Total Paradas Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para total paradas prioritários.</em></p>
    <img src='comparacao_total_paradas_prioritarios_qlearning.png' alt='Total Paradas Prioritários - Q-Learning'>
    
    <h3>Tempo Espera Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para tempo espera prioritários.</em></p>
    <img src='comparacao_tempo_espera_prioritarios_qlearning.png' alt='Tempo Espera Prioritários - Q-Learning'>
    
    <h3>Velocidade Média Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    <p><em>Gráfico mostra a evolução ao longo do tempo para velocidade média prioritários.</em></p>
    <img src='comparacao_velocidade_media_prioritarios_qlearning.png' alt='Velocidade Média Prioritários - Q-Learning'>
</body>
</html>
""".format(
    f"{metrics_rl['tempo_espera_emergency']['media']:.2f}",
    f"{metrics_rl['tempo_espera_emergency']['desvio_padrao']:.2f}",
    f"{metrics_rl['tempo_espera_emergency']['maximo']:.2f}",
    f"{metrics_rl['tempo_espera_emergency']['minimo']:.2f}",
    f"{metrics_rl['tempo_espera_authority']['media']:.2f}",
    f"{metrics_rl['tempo_espera_authority']['desvio_padrao']:.2f}",
    f"{metrics_rl['tempo_espera_authority']['maximo']:.2f}",
    f"{metrics_rl['tempo_espera_authority']['minimo']:.2f}",
    f"{metrics_rl['carros_parados_prioritarios']['media']:.2f}",
    f"{metrics_rl['carros_parados_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_rl['carros_parados_prioritarios']['maximo']:.2f}",
    f"{metrics_rl['carros_parados_prioritarios']['minimo']:.2f}",
    f"{metrics_rl['total_paradas_prioritarios']['media']:.2f}",
    f"{metrics_rl['total_paradas_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_rl['total_paradas_prioritarios']['maximo']:.2f}",
    f"{metrics_rl['total_paradas_prioritarios']['minimo']:.2f}",
    f"{metrics_rl['tempo_espera_prioritarios']['media']:.2f}",
    f"{metrics_rl['tempo_espera_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_rl['tempo_espera_prioritarios']['maximo']:.2f}",
    f"{metrics_rl['tempo_espera_prioritarios']['minimo']:.2f}",
    f"{metrics_rl['velocidade_media_prioritarios']['media']:.2f}",
    f"{metrics_rl['velocidade_media_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_rl['velocidade_media_prioritarios']['maximo']:.2f}",
    f"{metrics_rl['velocidade_media_prioritarios']['minimo']:.2f}"
)

with open("relatorio/relatorio_prioritarios_qlearning.html", "w") as f:
    f.write(html_rl)

# Gerar relatório comparativo para Prioritários
html_prioritarios_comparativo = """
<html>
<head>
    <title>Relatório Comparativo de Veículos Prioritários: Tempo Fixo vs Q-Learning</title>
</head>
<body>
    <h1>Relatório Comparativo de Veículos Prioritários: Tempo Fixo vs Q-Learning</h1>
    <p>Este relatório compara o desempenho dos algoritmos Tempo Fixo e Q-Learning especificamente para veículos prioritários (emergência e autoridade).</p>
    
    <h2>Métricas para Tempo de Espera - Emergência</h2>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    
    <h2>Métricas para Tempo de Espera - Autoridade</h2>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    
    <h2>Métricas Gerais para Veículos Prioritários</h2>
    <h3>Carros Parados Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    
    <h3>Total Paradas Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    
    <h3>Tempo Espera Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    
    <h3>Velocidade Média Prioritários</h3>
    <table border='1' cellpadding='5'>
      <tr><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Tempo Fixo</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
      <tr><td>Q-Learning</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    
    <h2>Comparação de Médias - Prioritários</h2>
    <img src='comparacao_medias_prioritarios.png' alt='Comparação de Médias - Prioritários'>
</body>
</html>
""".format(
    f"{metrics_fixed['tempo_espera_emergency']['media']:.2f}",
    f"{metrics_fixed['tempo_espera_emergency']['desvio_padrao']:.2f}",
    f"{metrics_fixed['tempo_espera_emergency']['maximo']:.2f}",
    f"{metrics_fixed['tempo_espera_emergency']['minimo']:.2f}",
    f"{metrics_rl['tempo_espera_emergency']['media']:.2f}",
    f"{metrics_rl['tempo_espera_emergency']['desvio_padrao']:.2f}",
    f"{metrics_rl['tempo_espera_emergency']['maximo']:.2f}",
    f"{metrics_rl['tempo_espera_emergency']['minimo']:.2f}",
    f"{metrics_fixed['tempo_espera_authority']['media']:.2f}",
    f"{metrics_fixed['tempo_espera_authority']['desvio_padrao']:.2f}",
    f"{metrics_fixed['tempo_espera_authority']['maximo']:.2f}",
    f"{metrics_fixed['tempo_espera_authority']['minimo']:.2f}",
    f"{metrics_rl['tempo_espera_authority']['media']:.2f}",
    f"{metrics_rl['tempo_espera_authority']['desvio_padrao']:.2f}",
    f"{metrics_rl['tempo_espera_authority']['maximo']:.2f}",
    f"{metrics_rl['tempo_espera_authority']['minimo']:.2f}",
    f"{metrics_fixed['carros_parados_prioritarios']['media']:.2f}",
    f"{metrics_fixed['carros_parados_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_fixed['carros_parados_prioritarios']['maximo']:.2f}",
    f"{metrics_fixed['carros_parados_prioritarios']['minimo']:.2f}",
    f"{metrics_rl['carros_parados_prioritarios']['media']:.2f}",
    f"{metrics_rl['carros_parados_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_rl['carros_parados_prioritarios']['maximo']:.2f}",
    f"{metrics_rl['carros_parados_prioritarios']['minimo']:.2f}",
    f"{metrics_fixed['total_paradas_prioritarios']['media']:.2f}",
    f"{metrics_fixed['total_paradas_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_fixed['total_paradas_prioritarios']['maximo']:.2f}",
    f"{metrics_fixed['total_paradas_prioritarios']['minimo']:.2f}",
    f"{metrics_rl['total_paradas_prioritarios']['media']:.2f}",
    f"{metrics_rl['total_paradas_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_rl['total_paradas_prioritarios']['maximo']:.2f}",
    f"{metrics_rl['total_paradas_prioritarios']['minimo']:.2f}",
    f"{metrics_fixed['tempo_espera_prioritarios']['media']:.2f}",
    f"{metrics_fixed['tempo_espera_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_fixed['tempo_espera_prioritarios']['maximo']:.2f}",
    f"{metrics_fixed['tempo_espera_prioritarios']['minimo']:.2f}",
    f"{metrics_rl['tempo_espera_prioritarios']['media']:.2f}",
    f"{metrics_rl['tempo_espera_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_rl['tempo_espera_prioritarios']['maximo']:.2f}",
    f"{metrics_rl['tempo_espera_prioritarios']['minimo']:.2f}",
    f"{metrics_fixed['velocidade_media_prioritarios']['media']:.2f}",
    f"{metrics_fixed['velocidade_media_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_fixed['velocidade_media_prioritarios']['maximo']:.2f}",
    f"{metrics_fixed['velocidade_media_prioritarios']['minimo']:.2f}",
    f"{metrics_rl['velocidade_media_prioritarios']['media']:.2f}",
    f"{metrics_rl['velocidade_media_prioritarios']['desvio_padrao']:.2f}",
    f"{metrics_rl['velocidade_media_prioritarios']['maximo']:.2f}",
    f"{metrics_rl['velocidade_media_prioritarios']['minimo']:.2f}"
)

with open("relatorio/relatorio_comparativo_prioritarios.html", "w") as f:
    f.write(html_prioritarios_comparativo)

# Gerar relatório comparativo Geral vs Prioritários
html_geral_vs_prioritarios = """
<html>
<head>
    <title>Relatório Comparativo: Geral vs Prioritários</title>
</head>
<body>
    <h1>Relatório Comparativo: Resultados Gerais vs Prioritários</h1>
    <p>Este relatório compara as métricas gerais de todos os veículos com as métricas específicas de veículos prioritários, destacando melhorias no tratamento de emergências e autoridades.</p>
    
    <h2>Carros Parados</h2>
    <table border='1' cellpadding='5'>
      <tr><th>Categoria</th><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Geral</td><td>Tempo Fixo</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Geral</td><td>Q-Learning</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Prioritários</td><td>Tempo Fixo</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Prioritários</td><td>Q-Learning</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
    </table>
    <p><strong>Melhorias nos Prioritários:</strong> Tempo Fixo: {:.2f}% redução, Q-Learning: {:.2f}% redução (comparado ao geral).</p>
    
    <h2>Total de Paradas</h2>
    <table border='1' cellpadding='5'>
      <tr><th>Categoria</th><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Geral</td><td>Tempo Fixo</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Geral</td><td>Q-Learning</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Prioritários</td><td>Tempo Fixo</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Prioritários</td><td>Q-Learning</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
    </table>
    <p><strong>Melhorias nos Prioritários:</strong> Tempo Fixo: {:.2f}% redução, Q-Learning: {:.2f}% redução (comparado ao geral).</p>
    
    <h2>Tempo de Espera</h2>
    <table border='1' cellpadding='5'>
      <tr><th>Categoria</th><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Geral</td><td>Tempo Fixo</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Geral</td><td>Q-Learning</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Prioritários</td><td>Tempo Fixo</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Prioritários</td><td>Q-Learning</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
    </table>
    <p><strong>Melhorias nos Prioritários:</strong> Tempo Fixo: {:.2f}% redução, Q-Learning: {:.2f}% redução (comparado ao geral).</p>
    
    <h2>Velocidade Média</h2>
    <table border='1' cellpadding='5'>
      <tr><th>Categoria</th><th>Algoritmo</th><th>Média</th><th>Desvio Padrão</th><th>Máximo</th><th>Mínimo</th></tr>
      <tr><td>Geral</td><td>Tempo Fixo</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Geral</td><td>Q-Learning</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Prioritários</td><td>Tempo Fixo</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
      <tr><td>Prioritários</td><td>Q-Learning</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>
    </table>
    <p><strong>Melhorias nos Prioritários:</strong> Tempo Fixo: {:.2f}% aumento, Q-Learning: {:.2f}% aumento (comparado ao geral).</p>
    
    <h2>Comparação de Médias - Geral vs Prioritários</h2>
    <img src='comparacao_geral_vs_prioritarios.png' alt='Comparação Geral vs Prioritários'>
</body>
</html>
""".format(
    metrics_fixed['carros_parados']['media'], metrics_fixed['carros_parados']['desvio_padrao'], metrics_fixed['carros_parados']['maximo'], metrics_fixed['carros_parados']['minimo'],
    metrics_rl['carros_parados']['media'], metrics_rl['carros_parados']['desvio_padrao'], metrics_rl['carros_parados']['maximo'], metrics_rl['carros_parados']['minimo'],
    metrics_fixed['carros_parados_prioritarios']['media'], metrics_fixed['carros_parados_prioritarios']['desvio_padrao'], metrics_fixed['carros_parados_prioritarios']['maximo'], metrics_fixed['carros_parados_prioritarios']['minimo'],
    metrics_rl['carros_parados_prioritarios']['media'], metrics_rl['carros_parados_prioritarios']['desvio_padrao'], metrics_rl['carros_parados_prioritarios']['maximo'], metrics_rl['carros_parados_prioritarios']['minimo'],
    (metrics_fixed['carros_parados']['media'] - metrics_fixed['carros_parados_prioritarios']['media']) / metrics_fixed['carros_parados']['media'] * 100 if metrics_fixed['carros_parados']['media'] > 0 else 0,
    (metrics_rl['carros_parados']['media'] - metrics_rl['carros_parados_prioritarios']['media']) / metrics_rl['carros_parados']['media'] * 100 if metrics_rl['carros_parados']['media'] > 0 else 0,
    
    metrics_fixed['total_paradas']['media'], metrics_fixed['total_paradas']['desvio_padrao'], metrics_fixed['total_paradas']['maximo'], metrics_fixed['total_paradas']['minimo'],
    metrics_rl['total_paradas']['media'], metrics_rl['total_paradas']['desvio_padrao'], metrics_rl['total_paradas']['maximo'], metrics_rl['total_paradas']['minimo'],
    metrics_fixed['total_paradas_prioritarios']['media'], metrics_fixed['total_paradas_prioritarios']['desvio_padrao'], metrics_fixed['total_paradas_prioritarios']['maximo'], metrics_fixed['total_paradas_prioritarios']['minimo'],
    metrics_rl['total_paradas_prioritarios']['media'], metrics_rl['total_paradas_prioritarios']['desvio_padrao'], metrics_rl['total_paradas_prioritarios']['maximo'], metrics_rl['total_paradas_prioritarios']['minimo'],
    (metrics_fixed['total_paradas']['media'] - metrics_fixed['total_paradas_prioritarios']['media']) / metrics_fixed['total_paradas']['media'] * 100 if metrics_fixed['total_paradas']['media'] > 0 else 0,
    (metrics_rl['total_paradas']['media'] - metrics_rl['total_paradas_prioritarios']['media']) / metrics_rl['total_paradas']['media'] * 100 if metrics_rl['total_paradas']['media'] > 0 else 0,
    
    metrics_fixed['tempo_espera']['media'], metrics_fixed['tempo_espera']['desvio_padrao'], metrics_fixed['tempo_espera']['maximo'], metrics_fixed['tempo_espera']['minimo'],
    metrics_rl['tempo_espera']['media'], metrics_rl['tempo_espera']['desvio_padrao'], metrics_rl['tempo_espera']['maximo'], metrics_rl['tempo_espera']['minimo'],
    metrics_fixed['tempo_espera_prioritarios']['media'], metrics_fixed['tempo_espera_prioritarios']['desvio_padrao'], metrics_fixed['tempo_espera_prioritarios']['maximo'], metrics_fixed['tempo_espera_prioritarios']['minimo'],
    metrics_rl['tempo_espera_prioritarios']['media'], metrics_rl['tempo_espera_prioritarios']['desvio_padrao'], metrics_rl['tempo_espera_prioritarios']['maximo'], metrics_rl['tempo_espera_prioritarios']['minimo'],
    (metrics_fixed['tempo_espera']['media'] - metrics_fixed['tempo_espera_prioritarios']['media']) / metrics_fixed['tempo_espera']['media'] * 100 if metrics_fixed['tempo_espera']['media'] > 0 else 0,
    (metrics_rl['tempo_espera']['media'] - metrics_rl['tempo_espera_prioritarios']['media']) / metrics_rl['tempo_espera']['media'] * 100 if metrics_rl['tempo_espera']['media'] > 0 else 0,
    
    metrics_fixed['velocidade_media']['media'], metrics_fixed['velocidade_media']['desvio_padrao'], metrics_fixed['velocidade_media']['maximo'], metrics_fixed['velocidade_media']['minimo'],
    metrics_rl['velocidade_media']['media'], metrics_rl['velocidade_media']['desvio_padrao'], metrics_rl['velocidade_media']['maximo'], metrics_rl['velocidade_media']['minimo'],
    metrics_fixed['velocidade_media_prioritarios']['media'], metrics_fixed['velocidade_media_prioritarios']['desvio_padrao'], metrics_fixed['velocidade_media_prioritarios']['maximo'], metrics_fixed['velocidade_media_prioritarios']['minimo'],
    metrics_rl['velocidade_media_prioritarios']['media'], metrics_rl['velocidade_media_prioritarios']['desvio_padrao'], metrics_rl['velocidade_media_prioritarios']['maximo'], metrics_rl['velocidade_media_prioritarios']['minimo'],
    (metrics_fixed['velocidade_media_prioritarios']['media'] - metrics_fixed['velocidade_media']['media']) / metrics_fixed['velocidade_media']['media'] * 100 if metrics_fixed['velocidade_media']['media'] > 0 else 0,
    (metrics_rl['velocidade_media_prioritarios']['media'] - metrics_rl['velocidade_media']['media']) / metrics_rl['velocidade_media']['media'] * 100 if metrics_rl['velocidade_media']['media'] > 0 else 0
)

with open("relatorio/relatorio_geral_vs_prioritarios.html", "w") as f:
    f.write(html_geral_vs_prioritarios)

html_path = os.path.join(output_dir, 'relatorio_comparativo.html')
gerar_html(metrics_fixed, metrics_rl, html_path)

# Geração de relatório em PDF
def gerar_pdf(metrics_fixed, metrics_rl, output_file):
    c = pdfcanvas.Canvas(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    width, height = letter

    # Título
    c.setFont('Helvetica-Bold', 16)
    c.drawCentredString(width/2, height - 50, 'Relatório de Comparação: Tempo Fixo vs Q-Learning')
    
    # Introdução
    c.setFont('Helvetica', 12)
    intro_text = "Este relatório compara o controle de semáforos tradicional (tempo fixo) com aprendizado por reforço (Q-Learning). Valores menores em congestionamento indicam melhor desempenho."
    text_object = c.beginText(50, height - 80)
    text_object.setFont('Helvetica', 10)
    text_object.textLines(intro_text)
    c.drawText(text_object)

    y_position = height - 120
    for metric in ['carros_parados', 'total_paradas', 'tempo_espera', 'velocidade_media']:
        # Métricas
        text = c.beginText(50, y_position)
        text.setFont('Helvetica-Bold', 12)
        text.textLine(f'Métricas para {metric_labels[metric]}:')
        text.setFont('Helvetica', 10)
        for alg, metrics in [('Tempo Fixo', metrics_fixed[metric]), ('Q-Learning', metrics_rl[metric])]:
            text.textLine(f"{alg} - Média: {metrics['media']:.2f}, Desvio Padrão: {metrics['desvio_padrao']:.2f}, Máx: {metrics['maximo']}, Mín: {metrics['minimo']}")
        c.drawText(text)
        y_position -= 50

        # Gráfico
        img_path = os.path.join(output_dir, f'comparacao_{metric}.png')
        if os.path.exists(img_path):
            img = Image(img_path, width=400, height=200)
            img.drawOn(c, 60, y_position - 220)
            y_position -= 240

        if y_position < 300:
            c.showPage()
            y_position = height - 50

    # Gráfico de barras
    img = Image(bar_plot_path, width=400, height=200)
    img.drawOn(c, 60, y_position - 220)

    c.showPage()
    c.save()

# Geração de relatório em PDF comparativo para Prioritários
def gerar_pdf_comparativo_prioritarios(metrics_fixed, metrics_rl, output_file):
    c = pdfcanvas.Canvas(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    width, height = letter

    # Título
    c.setFont('Helvetica-Bold', 16)
    c.drawCentredString(width/2, height - 50, 'Relatório Comparativo de Veículos Prioritários: Tempo Fixo vs Q-Learning')
    
    # Introdução
    c.setFont('Helvetica', 12)
    intro_text = "Este relatório compara o desempenho dos algoritmos Tempo Fixo e Q-Learning especificamente para veículos prioritários (emergência e autoridade)."
    text_object = c.beginText(50, height - 80)
    text_object.setFont('Helvetica', 10)
    text_object.textLines(intro_text)
    c.drawText(text_object)

    y_position = height - 120
    prioritarios_metrics = ['tempo_espera_emergency', 'tempo_espera_authority', 'carros_parados_prioritarios', 'total_paradas_prioritarios', 'tempo_espera_prioritarios', 'velocidade_media_prioritarios']
    for metric in prioritarios_metrics:
        # Métricas
        text = c.beginText(50, y_position)
        text.setFont('Helvetica-Bold', 12)
        text.textLine(f'Métricas para {metric_labels[metric]}:')
        text.setFont('Helvetica', 10)
        for alg, metrics in [('Tempo Fixo', metrics_fixed[metric]), ('Q-Learning', metrics_rl[metric])]:
            text.textLine(f"{alg} - Média: {metrics['media']:.2f}, Desvio Padrão: {metrics['desvio_padrao']:.2f}, Máx: {metrics['maximo']:.2f}, Mín: {metrics['minimo']:.2f}")
        c.drawText(text)
        y_position -= 50

        # Gráfico comparativo
        img_path = os.path.join(output_dir, f'comparacao_{metric}.png')
        if os.path.exists(img_path):
            img = Image(img_path, width=400, height=200)
            img.drawOn(c, 60, y_position - 220)
            y_position -= 240

        if y_position < 300:
            c.showPage()
            y_position = height - 50

    # Gráfico de barras prioritários
    img = Image(bar_plot_prioritarios_path, width=500, height=300)
    img.drawOn(c, 30, y_position - 320)

    c.showPage()
    c.save()

pdf_path = os.path.join(output_dir, 'relatorio_comparativo.pdf')
gerar_pdf(metrics_fixed, metrics_rl, pdf_path)

# Geração de relatório em PDF para Prioritários - Tempo Fixo
def gerar_pdf_prioritarios_tempo_fixo(metrics_fixed, output_file):
    c = pdfcanvas.Canvas(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    width, height = letter

    # Título
    c.setFont('Helvetica-Bold', 16)
    c.drawCentredString(width/2, height - 50, 'Relatório de Tratamento de Veículos Prioritários - Tempo Fixo')
    
    # Introdução
    c.setFont('Helvetica', 12)
    intro_text = "Este relatório apresenta métricas de desempenho para veículos prioritários (emergência e autoridade) no controle de semáforos com tempo fixo."
    text_object = c.beginText(50, height - 80)
    text_object.setFont('Helvetica', 10)
    text_object.textLines(intro_text)
    c.drawText(text_object)

    y_position = height - 120
    prioritarios_metrics = ['tempo_espera_emergency', 'tempo_espera_authority', 'carros_parados_prioritarios', 'total_paradas_prioritarios', 'tempo_espera_prioritarios', 'velocidade_media_prioritarios']
    for metric in prioritarios_metrics:
        # Métricas
        text = c.beginText(50, y_position)
        text.setFont('Helvetica-Bold', 12)
        text.textLine(f'Métricas para {metric_labels[metric]}:')
        text.setFont('Helvetica', 10)
        metrics = metrics_fixed[metric]
        text.textLine(f"Tempo Fixo - Média: {metrics['media']:.2f}, Desvio Padrão: {metrics['desvio_padrao']:.2f}, Máx: {metrics['maximo']:.2f}, Mín: {metrics['minimo']:.2f}")
        c.drawText(text)
        y_position -= 30

        # Gráfico
        img_path = os.path.join(output_dir, f'comparacao_{metric}_tempo_fixo.png')
        if os.path.exists(img_path):
            img = Image(img_path, width=400, height=200)
            img.drawOn(c, 60, y_position - 220)
            y_position -= 240

        if y_position < 300:
            c.showPage()
            y_position = height - 50

    c.save()

# Geração de relatório em PDF para Prioritários - Q-Learning
def gerar_pdf_prioritarios_qlearning(metrics_rl, output_file):
    c = pdfcanvas.Canvas(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    width, height = letter

    # Título
    c.setFont('Helvetica-Bold', 16)
    c.drawCentredString(width/2, height - 50, 'Relatório de Tratamento de Veículos Prioritários - Q-Learning')
    
    # Introdução
    c.setFont('Helvetica', 12)
    intro_text = "Este relatório apresenta métricas de desempenho para veículos prioritários (emergência e autoridade) no controle de semáforos com Q-Learning."
    text_object = c.beginText(50, height - 80)
    text_object.setFont('Helvetica', 10)
    text_object.textLines(intro_text)
    c.drawText(text_object)

    y_position = height - 120
    prioritarios_metrics = ['tempo_espera_emergency', 'tempo_espera_authority', 'carros_parados_prioritarios', 'total_paradas_prioritarios', 'tempo_espera_prioritarios', 'velocidade_media_prioritarios']
    for metric in prioritarios_metrics:
        # Métricas
        text = c.beginText(50, y_position)
        text.setFont('Helvetica-Bold', 12)
        text.textLine(f'Métricas para {metric_labels[metric]}:')
        text.setFont('Helvetica', 10)
        metrics = metrics_rl[metric]
        text.textLine(f"Q-Learning - Média: {metrics['media']:.2f}, Desvio Padrão: {metrics['desvio_padrao']:.2f}, Máx: {metrics['maximo']:.2f}, Mín: {metrics['minimo']:.2f}")
        c.drawText(text)
        y_position -= 30

        # Gráfico
        img_path = os.path.join(output_dir, f'comparacao_{metric}_qlearning.png')
        if os.path.exists(img_path):
            img = Image(img_path, width=400, height=200)
            img.drawOn(c, 60, y_position - 220)
            y_position -= 240

        if y_position < 300:
            c.showPage()
            y_position = height - 50

    c.save()

pdf_prioritarios_fixed_path = os.path.join(output_dir, 'relatorio_prioritarios_tempo_fixo.pdf')
gerar_pdf_prioritarios_tempo_fixo(metrics_fixed, pdf_prioritarios_fixed_path)

pdf_prioritarios_rl_path = os.path.join(output_dir, 'relatorio_prioritarios_qlearning.pdf')
gerar_pdf_prioritarios_qlearning(metrics_rl, pdf_prioritarios_rl_path)

pdf_comparativo_prioritarios_path = os.path.join(output_dir, 'relatorio_comparativo_prioritarios.pdf')
gerar_pdf_comparativo_prioritarios(metrics_fixed, metrics_rl, pdf_comparativo_prioritarios_path)

# Geração de relatório em PDF Geral vs Prioritários
def gerar_pdf_geral_vs_prioritarios(metrics_fixed, metrics_rl, output_file):
    c = pdfcanvas.Canvas(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    width, height = letter

    # Título
    c.setFont('Helvetica-Bold', 16)
    c.drawCentredString(width/2, height - 50, 'Relatório Comparativo: Geral vs Prioritários')
    
    # Introdução
    c.setFont('Helvetica', 12)
    intro_text = "Este relatório compara as métricas gerais de todos os veículos com as métricas específicas de veículos prioritários, destacando melhorias no tratamento de emergências e autoridades."
    text_object = c.beginText(50, height - 80)
    text_object.setFont('Helvetica', 10)
    text_object.textLines(intro_text)
    c.drawText(text_object)

    y_position = height - 120
    metrics_list = ['carros_parados', 'total_paradas', 'tempo_espera', 'velocidade_media']
    for metric in metrics_list:
        # Métricas Geral
        text = c.beginText(50, y_position)
        text.setFont('Helvetica-Bold', 12)
        text.textLine(f'Métricas Gerais para {metric_labels[metric]}:')
        text.setFont('Helvetica', 10)
        for alg, metrics in [('Tempo Fixo', metrics_fixed[metric]), ('Q-Learning', metrics_rl[metric])]:
            text.textLine(f"{alg} - Média: {metrics['media']:.2f}, Desvio Padrão: {metrics['desvio_padrao']:.2f}, Máx: {metrics['maximo']:.2f}, Mín: {metrics['minimo']:.2f}")
        c.drawText(text)
        y_position -= 30

        # Métricas Prioritários
        text = c.beginText(50, y_position)
        text.setFont('Helvetica-Bold', 12)
        text.textLine(f'Métricas Prioritários para {metric_labels[metric]}:')
        text.setFont('Helvetica', 10)
        prioritario_metric = metric + '_prioritarios' if metric != 'velocidade_media' else 'velocidade_media_prioritarios'
        for alg, metrics in [('Tempo Fixo', metrics_fixed[prioritario_metric]), ('Q-Learning', metrics_rl[prioritario_metric])]:
            text.textLine(f"{alg} - Média: {metrics['media']:.2f}, Desvio Padrão: {metrics['desvio_padrao']:.2f}, Máx: {metrics['maximo']:.2f}, Mín: {metrics['minimo']:.2f}")
        c.drawText(text)
        y_position -= 30

        # Melhorias
        if metric in ['carros_parados', 'total_paradas', 'tempo_espera']:
            melhoria_fixed = (metrics_fixed[metric]['media'] - metrics_fixed[prioritario_metric]['media']) / metrics_fixed[metric]['media'] * 100 if metrics_fixed[metric]['media'] > 0 else 0
            melhoria_rl = (metrics_rl[metric]['media'] - metrics_rl[prioritario_metric]['media']) / metrics_rl[metric]['media'] * 100 if metrics_rl[metric]['media'] > 0 else 0
            text.textLine(f"Melhorias: Tempo Fixo: {melhoria_fixed:.2f}% redução, Q-Learning: {melhoria_rl:.2f}% redução")
        else:
            melhoria_fixed = (metrics_fixed[prioritario_metric]['media'] - metrics_fixed[metric]['media']) / metrics_fixed[metric]['media'] * 100 if metrics_fixed[metric]['media'] > 0 else 0
            melhoria_rl = (metrics_rl[prioritario_metric]['media'] - metrics_rl[metric]['media']) / metrics_rl[metric]['media'] * 100 if metrics_rl[metric]['media'] > 0 else 0
            text.textLine(f"Melhorias: Tempo Fixo: {melhoria_fixed:.2f}% aumento, Q-Learning: {melhoria_rl:.2f}% aumento")
        c.drawText(text)
        y_position -= 20

        if y_position < 200:
            c.showPage()
            y_position = height - 50

    # Gráfico de barras geral vs prioritários
    img = Image(bar_plot_geral_vs_prior_path, width=600, height=300)
    img.drawOn(c, 20, y_position - 320)

    c.showPage()
    c.save()

pdf_geral_vs_prioritarios_path = os.path.join(output_dir, 'relatorio_geral_vs_prioritarios.pdf')
gerar_pdf_geral_vs_prioritarios(metrics_fixed, metrics_rl, pdf_geral_vs_prioritarios_path)

print(f"Relatórios gerados em: {output_dir}")

