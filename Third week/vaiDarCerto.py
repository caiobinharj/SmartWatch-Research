import pandas as pd
from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

#Processamento dos dataframes

#configuração 1 de dataframe: atividade, std_dev, absol_dev, resultant
#configuração 2 de dataframe: atividade, std_dev, resultant
#configuração 3 de dataframe: atividade, absol_dev, resultant
#configuração 4 de dataframe: atividade, resultant

# INÍCIO DO PROCESSAMENTO DE DADOS 

model_df1 = pd.DataFrame()
model_df2 = pd.DataFrame()
model_df3 = pd.DataFrame()
model_df4 = pd.DataFrame()
df_aux = pd.DataFrame()
df_cheio = pd.DataFrame()

# principais grupos de colunas 
standdev_cols = ['XSTANDDEV_x', 'YSTANDDEV_x', 'ZSTANDDEV_x', 
                 'XSTANDDEV_y', 'YSTANDDEV_y', 'ZSTANDDEV_y']

absoldev_cols = ['XABSOLDEV_x', 'YABSOLDEV_x', 'ZABSOLDEV_x', 
                 'XABSOLDEV_y', 'YABSOLDEV_y', 'ZABSOLDEV_y']

resultant_cols = ['RESULTANT_x', 'RESULTANT_y']

activity_cols = ['ACTIVITY']

print("Iniciando carregamento dos dados...")
for i in range(1600, 1651):
    
    if(i==1614 or i==1637 or i==1638 or i==1639 or i==1640):
        continue

    path_gyro = fr"C:\Users\caioa\Documents\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\arff_files\watch\gyro\data_{i}_gyro_watch.arff"
    path_accel = fr"C:\Users\caioa\Documents\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\arff_files\watch\accel\data_{i}_accel_watch.arff"

    data_gyro, meta_gyro = arff.loadarff(path_gyro)
    data_accel, meta_accel = arff.loadarff(path_accel)
        
    df_gyro = pd.DataFrame(data_gyro)
    df_accel = pd.DataFrame(data_accel)
    
    # Validação de dados: o accel tem um a mais de linha em: 1617, 1641, 1643, 1646
    if len(df_accel) != len(df_gyro):
        print(f"Arquivos {i} têm tamanhos diferentes. Pulando.")
        continue
        
    df_accel.columns = df_accel.columns.str.strip('"')
    df_gyro.columns = df_gyro.columns.str.strip('"')

    # Merge baseado no índice (posição das linhas)
    df_aux = pd.merge(df_accel, df_gyro, left_index=True, right_index=True, suffixes=('_x', '_y'))
    
    # Validação: Ver se as atividades batem (se não, a premissa do merge está errada)
    if not (df_aux['ACTIVITY_x'] == df_aux['ACTIVITY_y']).all():
        print(f"Atividades não correspondem no índice para o arquivo {i}. Pulando.")
        continue
        
    # Usar apenas uma coluna de atividade e renomear
    df_aux['ACTIVITY'] = df_aux['ACTIVITY_x']
    df_aux = df_aux.drop(columns=['ACTIVITY_x', 'ACTIVITY_y'])

    df_cheio = pd.concat([df_cheio, df_aux])

print("Carregamento de dados finalizado.")

# Resetar o índice depois do loop para evitar índices duplicados
df_cheio = df_cheio.reset_index(drop=True)

# Decodificar a coluna ACTIVITY de bytes para string (ex: b'A' -> 'A')

df_cheio['ACTIVITY'] = df_cheio['ACTIVITY'].str.decode('utf-8')

# Agora sim, crie os DataFrames de modelo
model_df1 = df_cheio[activity_cols+standdev_cols+absoldev_cols+resultant_cols]
model_df2 = df_cheio[activity_cols+standdev_cols+resultant_cols]
model_df3 = df_cheio[activity_cols+absoldev_cols+resultant_cols]
model_df4 = df_cheio[activity_cols+resultant_cols]
model_df5 = df_cheio[activity_cols+standdev_cols+absoldev_cols]


# E AGORA a coluna Tempo_segundos está correta
model_df1['Tempo_segundos'] = model_df1.index * 10
model_df2['Tempo_segundos'] = model_df2.index * 10
model_df3['Tempo_segundos'] = model_df3.index * 10
model_df4['Tempo_segundos'] = model_df4.index * 10
model_df5['Tempo_segundos'] = model_df5.index * 10


#Processamento de dataframes encerrado

#Treinamento dos K-Means

#Uso de Scaler para padraonizar os dados
scaler = StandardScaler()

#K-Means Configuração 1
scaled_data1 = scaler.fit_transform(model_df1.drop(['ACTIVITY', 'Tempo_segundos'], axis=1))
kmeans1 = KMeans(n_clusters=7, random_state=42) 
kmeans1.fit(scaled_data1)
model_df1['Cluster'] = kmeans1.labels_ 

#K-Means Configuração 2
scaled_data2 = scaler.fit_transform(model_df2.drop(['ACTIVITY', 'Tempo_segundos'], axis=1))
kmeans2 = KMeans(n_clusters=7, random_state=42)
kmeans2.fit(scaled_data2)
model_df2['Cluster'] = kmeans2.labels_

#K-Means Configuração 3
scaled_data3 = scaler.fit_transform(model_df3.drop(['ACTIVITY', 'Tempo_segundos'], axis=1))
kmeans3 = KMeans(n_clusters=7, random_state=42)
kmeans3.fit(scaled_data3)
model_df3['Cluster'] = kmeans3.labels_

#K-Means Configuração 4
#Dúvida: Scale desnecessário???
#Apesar de só serem duas colunas, vem de dados diferentes (accel e gyro)
scaled_data4 = scaler.fit_transform(model_df4.drop(['ACTIVITY', 'Tempo_segundos'], axis=1))
kmeans4 = KMeans(n_clusters=7, random_state=42)
kmeans4.fit(scaled_data4)
model_df4['Cluster'] = kmeans4.labels_

#K-Means Configuração 5
scaled_data9 = scaler.fit_transform(model_df5.drop(['ACTIVITY', 'Tempo_segundos'], axis=1))
kmeans4 = KMeans(n_clusters=7, random_state=42)
kmeans4.fit(scaled_data9)
model_df5['Cluster'] = kmeans4.labels_

#Treino dos Modelos finalizado
#Hora de plotar os gráficos


#Redimensionalização PCA para as reduções de dimensionalidade necessárias p/ plotagem de gráficos

#PCA1 transformação config1 (20  colunas) em uma dimensão
#PCA2 transformação config2 (8 colunas) em uma dimensão
#PCA3 transformação config3 (8 colunas) em uma dimensão
#PCA4 transformação config4 (2 colunas) em uma dimensão
#PCA5 transformação STANDDEV de df2 (6 colunas) em uma dimensão
#PCA6 transformação RESULTANT de df2 (2 colunas) em uma dimensão
#PCA7 transformação ABSOLDEV de df3 (6 colunas) em uma dimensão
#PCA8 transformação RESULTANT de df3 (2colunas) em uma dimensão

pca = PCA(n_components=1)


scaled_data5 = scaler.fit_transform(model_df2[standdev_cols])
scaled_data6 = scaler.fit_transform(model_df2[resultant_cols])
scaled_data7 = scaler.fit_transform(model_df3[absoldev_cols])
scaled_data8 = scaler.fit_transform(model_df3[resultant_cols])
scaled_data9 = scaler.fit_transform(model_df5[standdev_cols])
scaled_data10 = scaler.fit_transform(model_df5[absoldev_cols])


model_df1['PCA1'] = pca.fit_transform(scaled_data1)
model_df2['PCA2'] = pca.fit_transform(scaled_data2)
model_df3['PCA3'] = pca.fit_transform(scaled_data3)
model_df4['PCA4'] = pca.fit_transform(scaled_data4)
model_df2['PCA5'] = pca.fit_transform(scaled_data5)
model_df2['PCA6'] = pca.fit_transform(scaled_data6)
model_df3['PCA7'] = pca.fit_transform(scaled_data7)
model_df3['PCA8'] = pca.fit_transform(scaled_data8)
model_df5['PCA9'] = pca.fit_transform(scaled_data9)
model_df5['PCA10'] = pca.fit_transform(scaled_data10)


#Objeto das Cores

cores = {
    # Grupo Parado (Tons de Azul)
    'D': '#0dff00',  # Sitting
    'E': "#0dff00",  # Standing
    
    # Grupo Movimento Sutil (Tons de Amarelo/Laranja/Marrom)
    'F': '#000000',  # Typing
    'G': '#000000',  # Brushing Teeth
    'H': '#000000',  # Eating Soup
    'I': '#000000',  # Eating Chips
    'J': '#000000',  # Eating Pasta
    'K': '#000000',  # Drinking from Cup
    'L': '#000000',  # Eating Sandwich
    'Q': '#000000',  # Writing
    'R': '#000000',  # Clapping
    'S': '#000000',  # Folding Clothes
    
    # Grupo Movimento Intenso (Tons de Roxo/Magenta)
    'A': '#54278f',  # Walking
    'B': '#54278f',  # Jogging
    'C': '#54278f',  # Stairs
    'M': '#54278f',  # Kicking (Soccer Ball)
    'O': '#54278f',  # Playing Catch w/ Tennis Ball
    'P': '#54278f'   # Dribbling (Basketball)
}


#Gráfico 1: configuração 1 ((Standard Deviation, Absolute Deviation e Resultant) X Tempo) 
# (Seu dicionário 'cores' deve estar definido antes deste bloco)

# --- Gráfico 1: Estilo Classificação (Cluster vs Tempo) ---

# Aumentar o tamanho para melhor visualização (mais largo)
plt.figure(figsize=(18, 8)) 

# Iterar por cada ATIVIDADE REAL (do dicionário) para definir a COR
for activity_code, color in cores.items():
    
    # Filtrar o dataframe principal (model_df1) para esta atividade
    # model_df1 já contém 'ACTIVITY', 'Tempo_segundos' e 'Cluster'
    activity_data = model_df1[model_df1['ACTIVITY'] == activity_code]
    
    # Se essa atividade não existir nos dados carregados, pula
    if activity_data.empty:
        continue

    # Plotar como gráfico de dispersão (scatter plot)
    # Eixo X = Tempo (contínuo)
    # Eixo Y = Cluster (categórico, 0, 1 ou 2)
    # Cor = Cor da Atividade Real (do dicionário 'cores')
    plt.scatter(activity_data['Tempo_segundos'], 
                activity_data['Cluster'], 
                color=color,
                label=activity_code, 
                alpha=0.6, 
                s=15) # s = tamanho dos pontos

# Formatação do Gráfico 

plt.xlabel('Tempo', fontsize=12)
plt.ylabel('Cluster (Atribuído pelo K-Means)', fontsize=12)
plt.title('G1: Cluster (STDDEV, ABSOLDEV, RESULTANT) vs. Tempo', fontsize=15, fontweight='bold')

# Configurar o eixo Y para mostrar C0, C1, C2
cluster_ticks = sorted(model_df1['Cluster'].unique())
cluster_labels = [f'C{tick}' for tick in cluster_ticks]
plt.yticks(ticks=cluster_ticks, labels=cluster_labels, fontsize=10)

# Adicionar linhas de grade horizontais (como na sua imagem de exemplo)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mover a legenda para o lado de fora do gráfico
plt.legend(title='Atividade Real', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

plt.tight_layout() # Ajusta o layout para a legenda caber
plt.savefig('G1_Classificacao_Tempo_x_Cluster.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#Gráfico 2: configuração 2 ((Standard Deviation e Resultant) X Tempo) 
plt.figure(figsize=(18, 8)) 

for activity_code, color in cores.items():
    activity_data = model_df2[model_df2['ACTIVITY'] == activity_code]
    
    if activity_data.empty:
        continue

    plt.scatter(activity_data['Tempo_segundos'], 
                activity_data['Cluster'], 
                color=color,
                label=activity_code, 
                alpha=0.6, 
                s=15)

plt.xlabel('Tempo', fontsize=12)
plt.ylabel('Cluster (Atribuído pelo K-Means)', fontsize=12)
plt.title('G2: Cluster (STDDEV, RESULTANT) vs. Tempo', fontsize=15, fontweight='bold')

cluster_ticks = sorted(model_df2['Cluster'].unique())
cluster_labels = [f'C{tick}' for tick in cluster_ticks]
plt.yticks(ticks=cluster_ticks, labels=cluster_labels, fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Atividade Real', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout() 
plt.savefig('G2_Classificacao_Tempo_x_Cluster.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#Gráfico 3: configuração 3 ((Absolute Deviation e Resultant) X Tempo) 
plt.figure(figsize=(18, 8)) 

for activity_code, color in cores.items():
    activity_data = model_df3[model_df3['ACTIVITY'] == activity_code]
    
    if activity_data.empty:
        continue

    plt.scatter(activity_data['Tempo_segundos'], 
                activity_data['Cluster'], 
                color=color,
                label=activity_code, 
                alpha=0.6, 
                s=15)

plt.xlabel('Tempo', fontsize=12)
plt.ylabel('Cluster (Atribuído pelo K-Means)', fontsize=12)
plt.title('G3: Cluster (ABSOLDEV, RESULTANT) vs. Tempo', fontsize=15, fontweight='bold')

cluster_ticks = sorted(model_df3['Cluster'].unique())
cluster_labels = [f'C{tick}' for tick in cluster_ticks]
plt.yticks(ticks=cluster_ticks, labels=cluster_labels, fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Atividade Real', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout() 
plt.savefig('G3_Classificacao_Tempo_x_Cluster.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#Gráfico 4: configuração 4 (Resultant X Tempo) 
plt.figure(figsize=(18, 8)) 

for activity_code, color in cores.items():
    activity_data = model_df4[model_df4['ACTIVITY'] == activity_code]
    
    if activity_data.empty:
        continue

    plt.scatter(activity_data['Tempo_segundos'], 
                activity_data['Cluster'], 
                color=color,
                label=activity_code, 
                alpha=0.6, 
                s=15)

plt.xlabel('Tempo', fontsize=12)
plt.ylabel('Cluster (Atribuído pelo K-Means)', fontsize=12)
plt.title('G4: Cluster (RESULTANT) vs. Tempo', fontsize=15, fontweight='bold')

cluster_ticks = sorted(model_df4['Cluster'].unique())
cluster_labels = [f'C{tick}' for tick in cluster_ticks]
plt.yticks(ticks=cluster_ticks, labels=cluster_labels, fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Atividade Real', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout() 
plt.savefig('G4_Classificacao_Tempo_x_Cluster.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#Gráfico 5: Standard Deviation X Resultant (info talvez relevante: uso da configuração 2) 
plt.figure(figsize=(12, 8)) 

# Iterar por cada ATIVIDADE REAL para definir a COR
for activity_code, color in cores.items():
    activity_data = model_df2[model_df2['ACTIVITY'] == activity_code]
    
    if activity_data.empty:
        continue

    # Plotar como scatter: PCA5 (X) vs PCA6 (Y)
    plt.scatter(activity_data['PCA5'], 
                activity_data['PCA6'], 
                color=color,
                label=activity_code, 
                alpha=0.6, 
                s=30) # 's' é o tamanho do ponto

plt.xlabel('PCA5 (STDDEV)', fontsize=12)
plt.ylabel('PCA6 (RESULTANT)', fontsize=12)
plt.title('G5: realidade (STDDEV vs RESULTANT)', fontsize=15, fontweight='bold')

plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.legend(title='Atividade Real', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout() 
plt.savefig('G5_Realidade_PCA5_x_PCA6.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#Gráfico 6: Absolute Deviation X Resultant (info talvez relevante: uso da configuração 3) 
plt.figure(figsize=(12, 8)) 

for activity_code, color in cores.items():
    activity_data = model_df3[model_df3['ACTIVITY'] == activity_code]
    
    if activity_data.empty:
        continue

    plt.scatter(activity_data['PCA7'], 
                activity_data['PCA8'], 
                color=color,
                label=activity_code, 
                alpha=0.6, 
                s=30) 

plt.xlabel('PCA7 (ABSOLDEV)', fontsize=12)
plt.ylabel('PCA8 (RESULTANT)', fontsize=12)
plt.title('G6: realidade (ABSOLDEV X RESULTANT)', fontsize=15, fontweight='bold')

plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.legend(title='Atividade Real', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout() 
plt.savefig('G6_Realidade_PCA7_x_PCA8.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#Gráfico 7: Standard Deviation X Absolute Deviation (info talvez relevante: uso da configuração 5)
plt.figure(figsize=(12, 8)) 

for activity_code, color in cores.items():
    activity_data = model_df5[model_df5['ACTIVITY'] == activity_code]
    
    if activity_data.empty:
        continue

    plt.scatter(activity_data['PCA9'], 
                activity_data['PCA10'], 
                color=color,
                label=activity_code, 
                alpha=0.6, 
                s=30) 

plt.xlabel('PCA9 (STANDDEV)', fontsize=12)
plt.ylabel('PCA10 (ABSOLDEV)', fontsize=12)
plt.title('G7: realidade (STANDDEV X ABSOLDEV)', fontsize=15, fontweight='bold')

plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.legend(title='Atividade Real', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout() 
plt.savefig('G7_Realidade_PCA9_x_PCA10.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#Gráfico 0: (Standard Deviation, Absolute Deviation) X Tempo (info talvez relevante: uso da configuração 5)
plt.figure(figsize=(18, 8)) 

for activity_code, color in cores.items():
    activity_data = model_df2[model_df2['ACTIVITY'] == activity_code]
    
    if activity_data.empty:
        continue

    plt.scatter(activity_data['Tempo_segundos'], 
                activity_data['Cluster'], 
                color=color,
                label=activity_code, 
                alpha=0.6, 
                s=15)

plt.xlabel('Tempo', fontsize=12)
plt.ylabel('Cluster (Atribuído pelo K-Means)', fontsize=12)
plt.title('G0: Cluster (STDDEV, ABSOLDEV) vs. Tempo', fontsize=15, fontweight='bold')

cluster_ticks = sorted(model_df5['Cluster'].unique())
cluster_labels = [f'C{tick}' for tick in cluster_ticks]
plt.yticks(ticks=cluster_ticks, labels=cluster_labels, fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Atividade Real', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout() 
plt.savefig('G0_Classificacao_Tempo_x_Cluster.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()    



