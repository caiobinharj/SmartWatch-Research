"""
Script para gerar gráficos de clustering usando dados de acelerômetro e giroscópio.
Gera 12 gráficos: 3 modelos (K-means, DBSCAN, Hierárquico) x 4 features (RESULTANT, ABSOLDEV, STANDDEV, VAR)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.io import arff
import pandas as pd

# Configurações dos diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCEL_DIR = os.path.join(BASE_DIR, 'accel')
GYRO_DIR = os.path.join(BASE_DIR, 'gyro')
OUTPUT_DIR = os.path.join(BASE_DIR, 'graficos')

# Criar diretório de saída
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_arff_file(filepath):
    """
    Lê um arquivo ARFF e retorna os dados como DataFrame.
    
    Args:
        filepath: Caminho para o arquivo ARFF
        
    Returns:
        DataFrame com os dados ou None se houver erro
    """
    try:
        # Ler o arquivo e corrigir possíveis problemas na primeira linha
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Corrigir primeira linha se tiver caracteres estranhos (ex: 'x' no início)
        if lines and not lines[0].strip().startswith('@relation'):
            # Procurar pela linha @relation
            for i, line in enumerate(lines):
                if '@relation' in line:
                    lines[i] = line.replace('x', '').replace('  ', ' ').strip() + '\n'
                    break
        
        # Escrever arquivo temporário corrigido (scipy.io.arff precisa de formato correto)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.arff', delete=False, encoding='utf-8') as tmp:
            tmp.writelines(lines)
            tmp_path = tmp.name
        
        try:
            # Carregar ARFF usando scipy
            data, meta = arff.loadarff(tmp_path)
            df = pd.DataFrame(data)
            
            # Converter bytes para strings se necessário (atributos nominais vêm como bytes)
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except:
                        pass
            return df, meta
        finally:
            # Remover arquivo temporário
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        print(f"Erro ao ler {filepath}: {e}")
        return None, None

def extract_features_from_df(df, feature_type):
    """
    Extrai features específicas do DataFrame.
    
    Args:
        df: DataFrame com os dados do ARFF
        feature_type: Tipo de feature ('RESULTANT', 'ABSOLDEV', 'STANDDEV', 'VAR')
        
    Returns:
        Array numpy com as features ou None se não encontradas
    """
    # Remover aspas dos nomes das colunas se existirem (pandas pode manter aspas do ARFF)
    df.columns = df.columns.str.strip('"')
    
    if feature_type == 'RESULTANT':
        # RESULTANT: 1 dimensão (magnitude resultante)
        if 'RESULTANT' in df.columns:
            values = df[['RESULTANT']].values
            # Converter para float se necessário
            if values.dtype == object:
                values = pd.to_numeric(values.flatten(), errors='coerce').reshape(-1, 1)
            return values
        else:
            return None
    
    elif feature_type == 'ABSOLDEV':
        # ABSOLDEV: 3 dimensões (X, Y, Z absolute deviation)
        features = []
        for axis in ['X', 'Y', 'Z']:
            col_name = f'{axis}ABSOLDEV'
            if col_name in df.columns:
                values = df[col_name].values
                # Converter para float se necessário
                if values.dtype == object:
                    values = pd.to_numeric(values, errors='coerce')
                features.append(values)
        if len(features) == 3:
            return np.column_stack(features)
        return None
    
    elif feature_type == 'STANDDEV':
        # STANDDEV: 3 dimensões (X, Y, Z standard deviation)
        features = []
        for axis in ['X', 'Y', 'Z']:
            col_name = f'{axis}STANDDEV'
            if col_name in df.columns:
                values = df[col_name].values
                # Converter para float se necessário
                if values.dtype == object:
                    values = pd.to_numeric(values, errors='coerce')
                features.append(values)
        if len(features) == 3:
            return np.column_stack(features)
        return None
    
    elif feature_type == 'VAR':
        # VAR: 3 dimensões (X, Y, Z variance)
        features = []
        for axis in ['X', 'Y', 'Z']:
            col_name = f'{axis}VAR'
            if col_name in df.columns:
                values = df[col_name].values
                # Converter para float se necessário
                if values.dtype == object:
                    values = pd.to_numeric(values, errors='coerce')
                features.append(values)
        if len(features) == 3:
            return np.column_stack(features)
        return None
    
    return None

def load_all_data(file_range_start=1600, file_range_end=1650):
    """
    Carrega todos os dados dos arquivos de file_range_start a file_range_end.
    
    Args:
        file_range_start: Número do primeiro arquivo (padrão: 1600)
        file_range_end: Número do último arquivo (padrão: 1650)
        
    Returns:
        Tupla (all_accel_data, all_gyro_data) onde cada um é um dicionário
        com chave=file_num e valor=DataFrame
    """
    all_accel_data = {}
    all_gyro_data = {}
    missing_accel = []
    missing_gyro = []
    error_files = []
    
    # Tentar carregar cada arquivo no intervalo
    for file_num in range(file_range_start, file_range_end + 1):
        accel_file = os.path.join(ACCEL_DIR, f'data_{file_num}_accel_watch.arff')
        gyro_file = os.path.join(GYRO_DIR, f'data_{file_num}_gyro_watch.arff')
        
        accel_loaded = False
        gyro_loaded = False
        
        # Carregar accel
        if os.path.exists(accel_file):
            accel_df, _ = parse_arff_file(accel_file)
            if accel_df is not None:
                all_accel_data[file_num] = accel_df
                accel_loaded = True
            else:
                error_files.append(f'accel_{file_num}')
        else:
            missing_accel.append(file_num)
        
        # Carregar gyro
        if os.path.exists(gyro_file):
            gyro_df, _ = parse_arff_file(gyro_file)
            if gyro_df is not None:
                all_gyro_data[file_num] = gyro_df
                gyro_loaded = True
            else:
                error_files.append(f'gyro_{file_num}')
        else:
            missing_gyro.append(file_num)
        
        # Só usar se ambos foram carregados
        if not (accel_loaded and gyro_loaded):
            if file_num not in missing_accel and file_num not in missing_gyro:
                if not accel_loaded:
                    error_files.append(f'accel_{file_num}')
                if not gyro_loaded:
                    error_files.append(f'gyro_{file_num}')
    
    if missing_accel:
        print(f"  Arquivos accel não encontrados: {missing_accel}")
    if missing_gyro:
        print(f"  Arquivos gyro não encontrados: {missing_gyro}")
    if error_files:
        print(f"  Arquivos com erro ao carregar: {error_files}")
    
    return all_accel_data, all_gyro_data

def combine_features(accel_data, gyro_data, feature_type):
    """
    Combina features de accel e gyro para criar um ponto.
    
    Para RESULTANT: 2 dimensões (1 de accel + 1 de gyro)
    Para outras: 6 dimensões (3 de accel + 3 de gyro)
    
    Args:
        accel_data: DataFrame do acelerômetro
        gyro_data: DataFrame do giroscópio
        feature_type: Tipo de feature a extrair
        
    Returns:
        Array numpy com features combinadas ou None se houver erro
    """
    accel_features = extract_features_from_df(accel_data, feature_type)
    gyro_features = extract_features_from_df(gyro_data, feature_type)
    
    if accel_features is None:
        return None
    if gyro_features is None:
        return None
    
    # Garantir que têm o mesmo número de linhas (combinar linha N de accel com linha N de gyro)
    min_len = min(len(accel_features), len(gyro_features))
    if min_len == 0:
        return None
    
    accel_features = accel_features[:min_len]
    gyro_features = gyro_features[:min_len]
    
    # Combinar features
    if feature_type == 'RESULTANT':
        # 2 dimensões: uma de accel, uma de gyro
        # accel_features tem shape (N, 1), gyro_features tem shape (N, 1)
        combined = np.column_stack([accel_features.flatten(), gyro_features.flatten()])
    else:
        # 6 dimensões: 3 de accel (X, Y, Z) + 3 de gyro (X, Y, Z)
        # accel_features tem shape (N, 3), gyro_features tem shape (N, 3)
        combined = np.column_stack([accel_features, gyro_features])
    
    return combined

def prepare_all_data(all_accel_data, all_gyro_data, feature_type):
    """
    Prepara todos os dados combinados de todos os arquivos.
    
    Args:
        all_accel_data: Dicionário com DataFrames do acelerômetro
        all_gyro_data: Dicionário com DataFrames do giroscópio
        feature_type: Tipo de feature a extrair
        
    Returns:
        Array numpy com todos os pontos combinados ou None se houver erro
    """
    all_combined = []
    files_with_errors = []
    
    # Processar cada arquivo
    for file_num in sorted(all_accel_data.keys()):
        if file_num in all_gyro_data:
            combined = combine_features(all_accel_data[file_num], all_gyro_data[file_num], feature_type)
            if combined is not None and len(combined) > 0:
                all_combined.append(combined)
            else:
                files_with_errors.append(file_num)
        else:
            files_with_errors.append(file_num)
    
    if len(all_combined) == 0:
        if files_with_errors:
            print(f"    ⚠ Nenhum dado válido encontrado. Arquivos com problema: {files_with_errors[:5]}...")
        return None
    
    # Concatenar todos os dados de todos os arquivos
    final_data = np.vstack(all_combined)
    
    # Remover NaN e infinitos (dados inválidos)
    mask = ~(np.isnan(final_data).any(axis=1) | np.isinf(final_data).any(axis=1))
    final_data = final_data[mask]
    
    if len(final_data) == 0:
        print(f"    ⚠ Todos os dados foram removidos após filtrar NaN/infinitos")
        return None
    
    return final_data

def apply_kmeans(data, n_clusters=3):
    """
    Aplica K-means clustering.
    
    Args:
        data: Array numpy com os dados
        n_clusters: Número de clusters (padrão: 3)
        
    Returns:
        Tupla (labels, model) onde labels são os clusters atribuídos
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    return labels, kmeans

def apply_dbscan(data, eps=None, min_samples=5):
    """
    Aplica DBSCAN clustering.
    
    Args:
        data: Array numpy com os dados
        eps: Distância máxima entre pontos do mesmo cluster (None = calcular automaticamente)
        min_samples: Número mínimo de pontos para formar cluster
        
    Returns:
        Tupla (labels, model) onde labels são os clusters atribuídos (-1 = ruído)
    """
    if eps is None:
        # Calcular eps automaticamente baseado na distância média entre pontos
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(data)
        distances, _ = nbrs.kneighbors(data)
        distances = np.sort(distances, axis=0)
        distances = distances[:, min_samples-1]
        eps = np.percentile(distances, 50)  # Usar mediana das distâncias
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels, dbscan

def apply_hierarchical(data, n_clusters=3):
    """
    Aplica clustering hierárquico (Agglomerative Clustering).
    
    Args:
        data: Array numpy com os dados
        n_clusters: Número de clusters (padrão: 3)
        
    Returns:
        Tupla (labels, model) onde labels são os clusters atribuídos
    """
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(data)
    return labels, hierarchical

def plot_clusters_2d(data, labels, title, filename, centroids=None):
    """
    Plota clusters em 2D (para feature RESULTANT).
    
    Args:
        data: Array numpy com dados 2D
        labels: Array com labels dos clusters
        title: Título do gráfico
        filename: Nome do arquivo para salvar
        centroids: Array com centróides (opcional, apenas para K-means)
    """
    plt.figure(figsize=(10, 8))
    
    # Mapear labels para cores
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    # Separar labels de ruído dos clusters válidos
    valid_labels = [l for l in unique_labels if l != -1]
    cluster_names = ['Parado', 'Movimento Sutil', 'Movimento Intenso']
    
    for i, label in enumerate(unique_labels):
        if label == -1:  # DBSCAN noise points
            plt.scatter(data[labels == label, 0], data[labels == label, 1], 
                       c='black', marker='x', s=50, alpha=0.5, label='Ruído')
        else:
            # Mapear label para índice do cluster (0, 1, 2)
            cluster_idx = valid_labels.index(label) % 3
            cluster_name = cluster_names[cluster_idx]
            plt.scatter(data[labels == label, 0], data[labels == label, 1], 
                       c=[colors[i]], label=cluster_name, alpha=0.6, s=50)
    
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centróides')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Accel RESULTANT', fontsize=12)
    plt.ylabel('Gyro RESULTANT', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_classification_6d(data, labels, title, filename, centroids=None):
    """
    Plota classificação para dados 6D usando gráfico de barras mostrando
    a distribuição dos clusters por dimensão.
    
    Args:
        data: Array numpy com dados 6D
        labels: Array com labels dos clusters
        title: Título do gráfico
        filename: Nome do arquivo para salvar
        centroids: Array com centróides (opcional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Nomes das dimensões
    dim_names = ['X Accel', 'Y Accel', 'Z Accel', 'X Gyro', 'Y Gyro', 'Z Gyro']
    cluster_names = ['Parado', 'Movimento Sutil', 'Movimento Intenso']
    
    # Separar labels válidos
    unique_labels = np.unique(labels)
    valid_labels = [l for l in unique_labels if l != -1]
    
    # Para cada dimensão, criar um gráfico de barras
    for dim in range(6):
        row = dim // 3
        col = dim % 3
        ax = axes[row, col]
        
        # Calcular estatísticas por cluster
        cluster_stats = []
        cluster_labels_list = []
        
        for label in valid_labels:
            if label != -1:
                cluster_data = data[labels == label, dim]
                if len(cluster_data) > 0:
                    cluster_stats.append({
                        'mean': np.mean(cluster_data),
                        'std': np.std(cluster_data),
                        'count': len(cluster_data)
                    })
                    cluster_labels_list.append(label)
        
        # Criar gráfico de barras com erro
        if cluster_stats:
            means = [s['mean'] for s in cluster_stats]
            stds = [s['std'] for s in cluster_stats]
            cluster_indices = [valid_labels.index(l) % 3 for l in cluster_labels_list]
            cluster_names_plot = [cluster_names[i] for i in cluster_indices]
            
            x_pos = np.arange(len(cluster_stats))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                         color=plt.cm.Spectral(np.linspace(0, 1, len(cluster_stats))))
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cluster_names_plot, rotation=45, ha='right')
            ax.set_ylabel('Valor Médio')
            ax.set_title(f'{dim_names[dim]}')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Função principal que executa todo o processo:
    1. Carrega dados de 1600 a 1650
    2. Para cada feature e modelo, aplica clustering
    3. Gera 12 gráficos
    """
    print("="*60)
    print("GERAÇÃO DE GRÁFICOS DE CLUSTERING - WISDM DATASET")
    print("="*60)
    print("\nCarregando dados dos arquivos 1600 a 1650...")
    
    all_accel_data, all_gyro_data = load_all_data(1600, 1650)
    
    total_expected = (1650 - 1600 + 1) * 2  # 51 accel + 51 gyro = 102 arquivos
    total_accel_loaded = len(all_accel_data)
    total_gyro_loaded = len(all_gyro_data)
    total_loaded = total_accel_loaded + total_gyro_loaded
    
    print(f"\nArquivos esperados: {total_expected} (51 accel + 51 gyro)")
    print(f"Arquivos carregados: {total_loaded} ({total_accel_loaded} accel + {total_gyro_loaded} gyro)")
    print(f"Arquivos faltando: {total_expected - total_loaded}")
    
    if total_loaded == 0:
        print("ERRO: Nenhum arquivo foi carregado!")
        return
    
    # Configurações dos 12 gráficos
    graphs_config = [
        # Gráficos 1-3: RESULTANT (2D) - pode visualizar diretamente
        {'feature': 'RESULTANT', 'model': 'kmeans', 'plot_func': plot_clusters_2d, 'num': 1},
        {'feature': 'RESULTANT', 'model': 'hierarchical', 'plot_func': plot_clusters_2d, 'num': 2},
        {'feature': 'RESULTANT', 'model': 'dbscan', 'plot_func': plot_clusters_2d, 'num': 3},
        
        # Gráficos 4-6: ABSOLDEV (6D) - usar gráfico de classificação
        {'feature': 'ABSOLDEV', 'model': 'kmeans', 'plot_func': plot_classification_6d, 'num': 4},
        {'feature': 'ABSOLDEV', 'model': 'hierarchical', 'plot_func': plot_classification_6d, 'num': 5},
        {'feature': 'ABSOLDEV', 'model': 'dbscan', 'plot_func': plot_classification_6d, 'num': 6},
        
        # Gráficos 7-9: STANDDEV (6D) - usar gráfico de classificação
        {'feature': 'STANDDEV', 'model': 'kmeans', 'plot_func': plot_classification_6d, 'num': 7},
        {'feature': 'STANDDEV', 'model': 'hierarchical', 'plot_func': plot_classification_6d, 'num': 8},
        {'feature': 'STANDDEV', 'model': 'dbscan', 'plot_func': plot_classification_6d, 'num': 9},
        
        # Gráficos 10-12: VAR (6D) - usar gráfico de classificação
        {'feature': 'VAR', 'model': 'kmeans', 'plot_func': plot_classification_6d, 'num': 10},
        {'feature': 'VAR', 'model': 'hierarchical', 'plot_func': plot_classification_6d, 'num': 11},
        {'feature': 'VAR', 'model': 'dbscan', 'plot_func': plot_classification_6d, 'num': 12},
    ]
    
    print("\n" + "="*60)
    print("PROCESSANDO GRÁFICOS")
    print("="*60)
    
    for config in graphs_config:
        print(f"\n[Gráfico {config['num']}/12] {config['model'].upper()} com {config['feature']}...")
        
        # Preparar dados
        data = prepare_all_data(all_accel_data, all_gyro_data, config['feature'])
        
        if data is None or len(data) == 0:
            print(f"  ❌ Erro: Não foi possível preparar dados para {config['feature']}")
            # Debug: verificar se as colunas existem no primeiro arquivo
            if all_accel_data and all_gyro_data:
                first_file = sorted(all_accel_data.keys())[0]
                accel_df = all_accel_data[first_file]
                gyro_df = all_gyro_data[first_file]
                accel_cols = [c for c in accel_df.columns if config['feature'] in c.upper() or 
                             (config['feature'] == 'RESULTANT' and 'RESULTANT' in c.upper())]
                gyro_cols = [c for c in gyro_df.columns if config['feature'] in c.upper() or 
                            (config['feature'] == 'RESULTANT' and 'RESULTANT' in c.upper())]
                print(f"    Debug - Colunas encontradas no arquivo {first_file}:")
                print(f"      Accel: {accel_cols[:5]}...")
                print(f"      Gyro: {gyro_cols[:5]}...")
            continue
        
        print(f"  ✓ Dados preparados: {data.shape[0]} pontos, {data.shape[1]} dimensões")
        
        # Aplicar clustering
        if config['model'] == 'kmeans':
            labels, model = apply_kmeans(data, n_clusters=3)
            centroids = model.cluster_centers_ if hasattr(model, 'cluster_centers_') else None
        elif config['model'] == 'dbscan':
            labels, model = apply_dbscan(data, eps=None, min_samples=5)
            centroids = None
        elif config['model'] == 'hierarchical':
            labels, model = apply_hierarchical(data, n_clusters=3)
            centroids = None
        
        n_clusters_found = len([l for l in np.unique(labels) if l != -1])
        n_noise = np.sum(labels == -1) if -1 in labels else 0
        print(f"  ✓ Clusters encontrados: {n_clusters_found}" + (f" (Ruído: {n_noise} pontos)" if n_noise > 0 else ""))
        
        # Criar gráfico
        title = f"Gráfico {config['num']}: {config['model'].upper()} - {config['feature']}"
        filename = f"grafico_{config['num']:02d}_{config['model']}_{config['feature'].lower()}.png"
        
        config['plot_func'](data, labels, title, filename, centroids)
        print(f"  ✓ Gráfico salvo: {filename}")
    
    print("\n" + "="*60)
    print("✅ TODOS OS GRÁFICOS FORAM GERADOS COM SUCESSO!")
    print("="*60)
    print(f"Gráficos salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
