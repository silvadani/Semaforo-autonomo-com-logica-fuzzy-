"""
comparar_resultados_refatorado.py
Versão refatorada do gerador de gráficos comparativos entre Tempo Fixo e Q-Learning
Objetivo: gerar as métricas e gráficos solicitados pelo usuário (em português).

Saída:
 - Imagens PNG salvas em ./relatorio_graficos/
 - CSV resumo com métricas em ./relatorio_graficos/resumo_metricas.csv

Observações importantes:
 - O script calcula métricas que dependem de subtração (ex.: total_não_prior = total - prioritários).
 - Para velocidades e tempos médios por categoria, o script usa as colunas existentes. Se não existir um número de veículos por categoria por instante, o script fará estimativas e marcará como "APROXIMADA".
 - Recomenda-se ter logs por-veículo com uma coluna indicando "prioridade" (0/1) para obter medidas exatas.

Como usar:
 - Coloque os CSVs no mesmo diretório do script.
 - Execute: python comparar_resultados_refatorado.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Diretório de saída
OUTPUT_DIR = Path("relatorio_graficos")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mapas de arquivos esperados (nomes conforme seu repositório atual)
fixed_files = {
    'carros_parados': "resultado_tempo_fixo.csv",
    'total_paradas': "paradas_tempo_fixo.csv",
    'tempo_espera': "espera_tempo_fixo.csv",
    'velocidade_media': "velocidade_tempo_fixo.csv",
    'tempo_espera_prioritarios': "espera_prioritarios_tempo_fixo.csv",
    'carros_parados_prioritarios': "carros_parados_prioritarios_tempo_fixo.csv",
    'total_paradas_prioritarios': "paradas_prioritarios_tempo_fixo.csv",
    'velocidade_media_prioritarios': "velocidade_prioritarios_tempo_fixo.csv"
}
rl_files = {
    'carros_parados': "resultado_qlearning.csv",
    'total_paradas': "paradas_qlearning.csv",
    'tempo_espera': "espera_qlearning.csv",
    'velocidade_media': "velocidade_qlearning.csv",
    'tempo_espera_prioritarios': "espera_prioritarios_qlearning.csv",
    'carros_parados_prioritarios': "carros_parados_prioritarios_qlearning.csv",
    'total_paradas_prioritarios': "paradas_prioritarios_qlearning.csv",
    'velocidade_media_prioritarios': "velocidade_prioritarios_qlearning.csv"
}

# Leitura segura de CSVs (retorna DataFrame vazio caso arquivo não exista)
def safe_read_csv(path, **kwargs):
    try:
        df = pd.read_csv(path, **kwargs)
        return df
    except FileNotFoundError:
        print(f"Aviso: arquivo não encontrado: {path}; gerando DataFrame vazio.")
        return pd.DataFrame()

# Carrega todos os dataframes
dfs_fixed = {k: safe_read_csv(v) for k, v in fixed_files.items()}
dfs_rl = {k: safe_read_csv(v) for k, v in rl_files.items()}

# Função utilitária para renomear coluna de valor padrão para um nome comum
def standardize_column(df, candidates):
    """Procura uma das colunas candidatas e retorna uma só coluna renomeada para 'value'."""
    if df.empty:
        return df
    for c in candidates:
        if c in df.columns:
            return df.rename(columns={c: 'value'})
    # se não encontrou, tenta a segunda coluna
    if len(df.columns) >= 2:
        df2 = df.copy()
        # assume que a segunda coluna é a de interesse
        df2.columns = [df.columns[0], 'value']
        return df2
    return df

# Padroniza todos os DFs para ter colunas ['tempo','value'] onde for possível
def prepare_time_series(df, value_candidates=None):
    if df.empty:
        return df
    if 'tempo' not in df.columns and 'time' in df.columns:
        df = df.rename(columns={'time': 'tempo'})
    if 'tempo' not in df.columns:
        # se não existe tempo, assume índice como tempo
        df = df.reset_index().rename(columns={'index': 'tempo'})
    if value_candidates is None:
        # tenta heurísticas comuns
        value_candidates = [c for c in df.columns if c != 'tempo']
    df_std = df[['tempo'] + [c for c in value_candidates if c in df.columns]]
    # pega a primeira coluna de valor disponível
    if len(df_std.columns) >= 2:
        val_col = df_std.columns[1]
        df_std = df_std[['tempo', val_col]].rename(columns={val_col: 'value'})
    else:
        # retorna com apenas tempo
        df_std = df_std.rename(columns={df_std.columns[0]:'tempo'})
    # garante tipo numérico em tempo
    df_std['tempo'] = pd.to_numeric(df_std['tempo'], errors='coerce')
    df_std = df_std.dropna(subset=['tempo'])
    df_std = df_std.sort_values('tempo').reset_index(drop=True)
    return df_std

# Prepara todos os DFs padronizados
prepared = {}
for k, df in dfs_fixed.items():
    prepared[f'fixed_{k}'] = prepare_time_series(df)
for k, df in dfs_rl.items():
    prepared[f'rl_{k}'] = prepare_time_series(df)

# Função para fazer merge de todas as séries por tempo
from functools import reduce

def merge_on_tempo(dfs_dict):
    dfs = []
    for name, df in dfs_dict.items():
        if df.empty:
            continue
        colname = name
        df2 = df.copy()
        df2 = df2.rename(columns={'value': colname})
        dfs.append(df2)
    if not dfs:
        return pd.DataFrame()
    merged = reduce(lambda left, right: pd.merge(left, right, on='tempo', how='outer'), dfs)
    merged = merged.sort_values('tempo').reset_index(drop=True)
    return merged

merged_fixed = merge_on_tempo({f: prepared[f'fixed_{f}'] for f in fixed_files.keys()})
merged_rl = merge_on_tempo({f: prepared[f'rl_{f}'] for f in rl_files.keys()})

# Funções para cálculo de métricas pedidas
def compute_aggregated_metrics(merged):
    """Retorna um dicionário com metrics: medias e totais por colunas padronizadas."""
    m = {}
    # helper
    def mean_col(df, col):
        if col in df.columns:
            return float(df[col].dropna().mean())
        return np.nan
    def sum_col(df, col):
        if col in df.columns:
            return float(df[col].dropna().sum())
        return np.nan

    # Tempos médios
    m['tempo_medio_total'] = mean_col(merged, 'tempo_espera')
    m['tempo_medio_prioritarios'] = mean_col(merged, 'tempo_espera_prioritarios')
    # tentativa de estimativa para não-prioritários: se existir coluna total e prioritarios, subtrai as métricas de soma e re-deriva media
    if ('tempo_espera' in merged.columns) and ('tempo_espera_prioritarios' in merged.columns):
        m['tempo_medio_nao_prioritarios_estimado'] = (m['tempo_medio_total'] - m['tempo_medio_prioritarios'])
    else:
        m['tempo_medio_nao_prioritarios_estimado'] = np.nan

    # Paradas (totais)
    m['total_paradas_total'] = sum_col(merged, 'total_paradas')
    m['total_paradas_prioritarios'] = sum_col(merged, 'total_paradas_prioritarios')
    if pd.notna(m['total_paradas_total']) and pd.notna(m['total_paradas_prioritarios']):
        m['total_paradas_nao_prioritarios'] = m['total_paradas_total'] - m['total_paradas_prioritarios']
    else:
        m['total_paradas_nao_prioritarios'] = np.nan

    # Carros parados
    m['carros_parados_total'] = sum_col(merged, 'carros_parados')
    m['carros_parados_prioritarios'] = sum_col(merged, 'carros_parados_prioritarios')
    if pd.notna(m['carros_parados_total']) and pd.notna(m['carros_parados_prioritarios']):
        m['carros_parados_nao_prioritarios'] = m['carros_parados_total'] - m['carros_parados_prioritarios']
    else:
        m['carros_parados_nao_prioritarios'] = np.nan

    # Velocidades - médias diretas
    m['velocidade_media_total'] = mean_col(merged, 'velocidade_media')
    m['velocidade_media_prioritarios'] = mean_col(merged, 'velocidade_media_prioritarios')
    if ('velocidade_media' in merged.columns) and ('velocidade_media_prioritarios' in merged.columns):
        m['velocidade_media_nao_prioritarios_estimada'] = m['velocidade_media_total'] - m['velocidade_media_prioritarios']
    else:
        m['velocidade_media_nao_prioritarios_estimada'] = np.nan

    # total paradas medio de todos os veiculos (média das paradas por instante)
    if 'total_paradas' in merged.columns:
        m['paradas_media_por_instante'] = merged['total_paradas'].dropna().mean()
    else:
        m['paradas_media_por_instante'] = np.nan

    return m

metrics_fixed = compute_aggregated_metrics(merged_fixed)
metrics_rl = compute_aggregated_metrics(merged_rl)

# Monta um DataFrame resumo para salvar
summary = pd.DataFrame({
    'metrica': list(set(list(metrics_fixed.keys()) + list(metrics_rl.keys()))),
})
summary['fixed'] = summary['metrica'].map(lambda x: metrics_fixed.get(x, np.nan))
summary['qlearning'] = summary['metrica'].map(lambda x: metrics_rl.get(x, np.nan))
summary.to_csv(OUTPUT_DIR / 'resumo_metricas.csv', index=False)

# === Funções de plotagem ===

def save_bar_comparison(metric_name, label, fixed_val, rl_val, filename):
    plt.figure(figsize=(6,4))
    bars = plt.bar(['Tempo Fixo','Q-Learning'], [fixed_val, rl_val])
    plt.title(label)
    plt.ylabel(label)
    # anota valores
    for rect, val in zip(bars, [fixed_val, rl_val]):
        if pd.notna(val):
            try:
                plt.text(rect.get_x() + rect.get_width()/2, rect.get_height()*1.01, f"{val:.2f}", ha='center')
            except Exception:
                plt.text(rect.get_x() + rect.get_width()/2, rect.get_height()*1.01, str(val), ha='center')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()

def save_line_timeseries(df, column, title, filename, start_zero=True, step=1, interpolate_method='linear'):
    """
    Plota uma série temporal contínua com reindexação e interpolação.
    - df: DataFrame com colunas ['tempo','<column>']
    - column: nome da coluna de interesse
    - step: passo entre instantes de tempo (1 = todos os inteiros entre min e max)
    - interpolate_method: método passado para pandas.interpolate()
    """
    if df.empty or column not in df.columns:
        print(f"Pulando plot {title}: dados não disponíveis")
        return

    # copia e garante tipos
    df2 = df[['tempo', column]].copy()
    df2['tempo'] = pd.to_numeric(df2['tempo'], errors='coerce')
    df2 = df2.dropna(subset=['tempo']).sort_values('tempo').reset_index(drop=True)

    # se não sobrou nada depois de limpar
    if df2.empty:
        print(f"Pulando plot {title}: sem tempos válidos")
        return

    # alerta se os pontos estão esparsos
    try:
        density_ratio = len(df2) / max(1, (int(df2['tempo'].max()) - int(df2['tempo'].min())))
        if density_ratio < 0.25:
            print(f"Aviso: série '{title}' parece esparsa ({len(df2)} pontos entre {int(df2['tempo'].min())} e {int(df2['tempo'].max())}). Será feita interpolação para visualização.")
    except Exception:
        pass

    # define índice contínuo de tempo (inteiros) entre min e max
    tmin = int(df2['tempo'].min())
    tmax = int(df2['tempo'].max())
    full_t = np.arange(tmin, tmax + 1, step)

    # reindex usando 'tempo' como índice
    df2 = df2.set_index('tempo')
    # Se houver duplicatas de tempo, agregue tirando a média
    df2 = df2.groupby(df2.index).mean()

    # reindex para preencher todos os instantes e interpolar valores faltantes
    df_full = df2.reindex(full_t)

    # tenta interpolar valores numéricos
    df_full[column] = df_full[column].interpolate(method=interpolate_method, limit_direction='both')

    # usa ffill/bfill (não mais fillna(method=...)) para evitar FutureWarning
    df_full[column] = df_full[column].ffill().bfill()

    # plota linha contínua (sem marcar apenas pontos)
    plt.figure(figsize=(8,4))
    plt.plot(df_full.index, df_full[column], linestyle='-', marker=None)
    plt.title(title)
    plt.xlabel('tempo')
    plt.ylabel(title)

    if start_zero:
        ymin = 0
        ymax = df_full[column].max() if pd.notna(df_full[column].max()) else None
        if ymax is not None:
            plt.ylim(bottom=ymin, top=ymax * 1.05)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()

# === Gera os plots solicitados ===

# 1) Tempo médio de todos os veículos: Tempo Fixo vs Q-Learning
save_bar_comparison('tempo_medio_total', 'Tempo médio de todos os veículos (s)', metrics_fixed.get('tempo_medio_total', np.nan), metrics_rl.get('tempo_medio_total', np.nan), 'comparacao_tempo_medio_total_barras.png')

# 2) Tempo de espera médio dos prioritários: Q-Learning vs Tempo Fixo
save_bar_comparison('tempo_medio_prioritarios', 'Tempo de espera médio (prioritários) (s)', metrics_fixed.get('tempo_medio_prioritarios', np.nan), metrics_rl.get('tempo_medio_prioritarios', np.nan), 'comparacao_tempo_espera_prioritarios_barras.png')

# 3) Tempo de espera dos não prioritários: APROXIMADO (diferença de médias)
save_bar_comparison('tempo_medio_nao_prioritarios_estimado', 'Tempo de espera médio (nao-prioritários) (APROX.) (s)', metrics_fixed.get('tempo_medio_nao_prioritarios_estimado', np.nan), metrics_rl.get('tempo_medio_nao_prioritarios_estimado', np.nan), 'comparacao_tempo_espera_nao_prioritarios_barras.png')

# 4) Total de paradas dos prioritários (Q-Learning vs Tempo Fixo)
save_bar_comparison('total_paradas_prioritarios', 'Total de paradas - Prioritários', metrics_fixed.get('total_paradas_prioritarios', np.nan), metrics_rl.get('total_paradas_prioritarios', np.nan), 'comparacao_total_paradas_prioritarios_barras.png')

# 5) Total de paradas dos não prioritários (exato via subtração)
save_bar_comparison('total_paradas_nao_prioritarios', 'Total de paradas - Nao Prioritários', metrics_fixed.get('total_paradas_nao_prioritarios', np.nan), metrics_rl.get('total_paradas_nao_prioritarios', np.nan), 'comparacao_total_paradas_nao_prioritarios_barras.png')

# 6) Total de paradas medio de todos os veiculos (média por instante)
save_bar_comparison('paradas_media_por_instante', 'Paradas média por instante (todos veiculos)', metrics_fixed.get('paradas_media_por_instante', np.nan), metrics_rl.get('paradas_media_por_instante', np.nan), 'comparacao_paradas_media_barras.png')

# 7) Velocidade média dos prioritários (Q-Learning vs Tempo Fixo)
save_bar_comparison('velocidade_media_prioritarios', 'Velocidade média - Prioritários (m/s)', metrics_fixed.get('velocidade_media_prioritarios', np.nan), metrics_rl.get('velocidade_media_prioritarios', np.nan), 'comparacao_velocidade_media_prioritarios_barras.png')

# 8) Velocidade dos não prioritários (APROXIMADA por diferença de médias)
save_bar_comparison('velocidade_media_nao_prioritarios_estimada', 'Velocidade média - Nao Prioritários (APROX.) (m/s)', metrics_fixed.get('velocidade_media_nao_prioritarios_estimada', np.nan), metrics_rl.get('velocidade_media_nao_prioritarios_estimada', np.nan), 'comparacao_velocidade_media_nao_prioritarios_barras.png')

# 9) Velocidade média total dos veículos (comparativo)
save_bar_comparison('velocidade_media_total', 'Velocidade média - Total (m/s)', metrics_fixed.get('velocidade_media_total', np.nan), metrics_rl.get('velocidade_media_total', np.nan), 'comparacao_velocidade_media_total_barras.png')

# === Plots em linha (timeseries) - agora com interpolação para visualização contínua ===
if not merged_fixed.empty:
    save_line_timeseries(merged_fixed, 'velocidade_media', 'Velocidade média ao longo do tempo (Tempo Fixo)', 'timeseries_velocidade_tempo_fixo.png', start_zero=True, step=1, interpolate_method='linear')
    save_line_timeseries(merged_fixed, 'velocidade_media_prioritarios', 'Velocidade prioritários - Tempo Fixo', 'timeseries_velocidade_prior_tempo_fixo.png', start_zero=True, step=1, interpolate_method='linear')
    save_line_timeseries(merged_fixed, 'tempo_espera', 'Tempo de espera ao longo do tempo (Tempo Fixo)', 'timeseries_espera_tempo_fixo.png', start_zero=True, step=1, interpolate_method='linear')

if not merged_rl.empty:
    save_line_timeseries(merged_rl, 'velocidade_media', 'Velocidade média ao longo do tempo (Q-Learning)', 'timeseries_velocidade_qlearning.png', start_zero=True, step=1, interpolate_method='linear')
    save_line_timeseries(merged_rl, 'velocidade_media_prioritarios', 'Velocidade prioritários - Q-Learning', 'timeseries_velocidade_prior_qlearning.png', start_zero=True, step=1, interpolate_method='linear')
    save_line_timeseries(merged_rl, 'tempo_espera', 'Tempo de espera ao longo do tempo (Q-Learning)', 'timeseries_espera_qlearning.png', start_zero=True, step=1, interpolate_method='linear')

# Salva também um CSV com as métricas consolidadas para facilitar análise posterior
summary.to_csv(OUTPUT_DIR / 'resumo_metricas.csv', index=False)

print(f"Gráficos e resumo gerados em: {OUTPUT_DIR.resolve()}")
print("Notas: para não-prioritários as métricas de velocidade/tempo foram estimadas quando não havia contagem por categoria. Para valores exatos, forneça logs por-veículo com campo 'prioridade'.")
