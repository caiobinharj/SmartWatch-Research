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

model_df1 = pd.DataFrame()
model_df2 = pd.DataFrame()
model_df3 = pd.DataFrame()
model_df4 = pd.DataFrame()
df_aux = pd.DataFrame()
df_cheio = pd.DataFrame()

for i in range(1600, 1651):
    
    #condição para verificar se é 1614, pq 1614 não existe.
    #1637, 1638, 1639 e 1640 tem muito mais accel que gyro.
    if(i==1614 or i==1637 or i==1638 or i==1639 or i==1640):
        continue

    #Abrindo o arquivo de acelerômetro e giroscópio do indivíduo indicado pelo id da iteração
    path_gyro = fr"C:\Users\caioa\Documents\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\arff_files\watch\gyro\data_{i}_gyro_watch.arff"
    path_accel = fr"C:\Users\caioa\Documents\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\arff_files\watch\accel\data_{i}_accel_watch.arff"

    #usando método loadarff do scipy p/ carregar arquivos .arff
    data_gyro, meta_gyro = arff.loadarff(path_gyro)
    data_accel, meta_accel = arff.loadarff(path_accel)

    df_gyro = pd.DataFrame(data_gyro)
    df_accel = pd.DataFrame(data_accel)
    
    #só pra retirar a feiura dessas aspas duplas nos nomes das colunas
    df_accel.columns = df_accel.columns.str.strip('"')
    df_gyro.columns = df_gyro.columns.str.strip('"')

    #formação do dataframe único a partir do dataframe auxiliar
    df_aux = pd.merge(df_accel, df_gyro, on= 'ACTIVITY')
    df_cheio = pd.concat([df_cheio, df_aux])
    

#definição das features para cada dataframe de modelo e adição do tempo
model_df1 = df_cheio[['ACTIVITY', 'XABSOLDEV_x', 'YABSOLDEV_x', 
                     'ZABSOLDEV_x', 'XSTANDDEV_x', 'YSTANDDEV_x', 
                     'ZSTANDDEV_x', 'RESULTANT_x', 'XABSOLDEV_y', 
                     'YABSOLDEV_y', 'ZABSOLDEV_y', 'XSTANDDEV_y', 
                     'YSTANDDEV_y','ZSTANDDEV_y', 'RESULTANT_y']]
model_df1['Tempo_segundos'] = model_df1.index * 10


model_df2 = df_cheio[['ACTIVITY', 'XSTANDDEV_x', 'YSTANDDEV_x', 
                     'ZSTANDDEV_x', 'RESULTANT_x', 'XSTANDDEV_y', 
                     'YSTANDDEV_y','ZSTANDDEV_y', 'RESULTANT_y']]
model_df2['Tempo_segundos'] = model_df1.index * 10

model_df3 = df_cheio[['ACTIVITY', 'XABSOLDEV_x', 'YABSOLDEV_x', 
                     'ZABSOLDEV_x', 'RESULTANT_x', 'XABSOLDEV_y', 
                     'YABSOLDEV_y', 'ZABSOLDEV_y', 'RESULTANT_y']]

model_df3['Tempo_segundos'] = model_df1.index * 10

model_df4 = df_cheio[['ACTIVITY', 'RESULTANT_x', 'RESULTANT_y']]
model_df4['Tempo_segundos'] = model_df1.index * 10



#Processamento de dataframes encerrado
#Treinamento dos K-Means

#Uso de Scaler para padraonizar os dados
scaler = StandardScaler()

#K-Means Configuração 1
scaled_data1 = scaler.fit_transform(model_df1.drop(['ACTIVITY', 'Tempo_segundos'], axis=1))
kmeans1 = KMeans(n_clusters=3, random_state=42) 
kmeans1.fit(scaled_data1)
model_df1['Cluster'] = kmeans1.labels_ 

#K-Means Configuração 2
scaled_data2 = scaler.fit_transform(model_df2.drop(['ACTIVITY', 'Tempo_segundos'], axis=1))
kmeans2 = KMeans(n_clusters=3, random_state=42)
kmeans2.fit(scaled_data2)
model_df2['Cluster'] = kmeans2.labels_

#K-Means Configuração 3
scaled_data3 = scaler.fit_transform(model_df3.drop(['ACTIVITY', 'Tempo_segundos'], axis=1))
kmeans3 = KMeans(n_clusters=3, random_state=42)
kmeans3.fit(scaled_data3)
model_df3['Cluster'] = kmeans3.labels_

#K-Means Configuração 4
#Dúvida: Scale desnecessário???
#Apesar de só serem duas colunas, vem de dados diferentes (accel e gyro)
scaled_data4 = scaler.fit_transform(model_df4.drop(['ACTIVITY', 'Tempo_segundos'], axis=1))
kmeans4 = KMeans(n_clusters=3, random_state=42)
kmeans4.fit(scaled_data4)
model_df4['Cluster'] = kmeans4.labels_

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

standdev_cols = ['XSTANDDEV_x', 'YSTANDDEV_x', 'ZSTANDDEV_x', 
                 'XSTANDDEV_y', 'YSTANDDEV_y', 'ZSTANDDEV_y']

absoldev_cols = ['XABSOLDEV_x', 'YABSOLDEV_x', 'ZABSOLDEV_x', 
                 'XABSOLDEV_y', 'YABSOLDEV_y', 'ZABSOLDEV_y']

resultant_cols = ['RESULTANT_x', 'RESULTANT_y']

scaled_data5 = scaler.fit_transform(model_df2[standdev_cols])
scaled_data6 = scaler.fit_transform(model_df2[resultant_cols])
scaled_data7 = scaler.fit_transform(model_df3[absoldev_cols])
scaled_data8 = scaler.fit_transform(model_df3[resultant_cols])

model_df1['PCA1'] = pca.fit_transform(scaled_data1)
model_df2['PCA2'] = pca.fit_transform(scaled_data2)
model_df3['PCA3'] = pca.fit_transform(scaled_data3)
model_df4['PCA4'] = pca.fit_transform(scaled_data4)
model_df2['PCA5'] = pca.fit_transform(scaled_data5)
model_df2['PCA6'] = pca.fit_transform(scaled_data6)
model_df3['PCA7'] = pca.fit_transform(scaled_data7)
model_df3['PCA8'] = pca.fit_transform(scaled_data8)



#Gráfico 1: configuração 1 ((Standard Deviation, Absolute Deviation e Resultant) X Tempo)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=model_df1, x='PCA1', y='Tempo_segundos', hue='Cluster', palette='viridis', alpha=0.7)
plt.title('G1: (STDDEV, ABSOLDEV, RESULTANT) X TEMPO')
plt.savefig('G1: PCA1XTEMPO.png')
plt.close()

#Gráfico 2: configuração 2 ((Standard Deviation e Resultant) X Tempo)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=model_df2, x='PCA2', y='Tempo_segundos', hue='Cluster', palette='viridis', alpha=0.7)
plt.title('G2: (STDDEV, RESULTANT) X TEMPO')
plt.savefig('G2: PCA2XTEMPO.png')
plt.close()

#Gráfico 3: configuração 3 ((Absolute Deviation e Resultant) X empo)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=model_df3, x='PCA3', y='Tempo_segundos', hue='Cluster', palette='viridis', alpha=0.7)
plt.title('G3: (ABSOLDEV, RESULTANT) X TEMPO')
plt.savefig('G3: PCA3XTEMPO.png')
plt.close()

#Gráfico 4: configuração 4 (Resultant X Tempo)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=model_df4, x='PCA4', y='Tempo_segundos', hue='Cluster', palette='viridis', alpha=0.7)
plt.title('G4: RESULTANT X TEMPO')
plt.savefig('G4: PCA4XTEMPO.png')
plt.close()

#Gráfico 5: Standard Deviation X Resultant (info talvez relevante: uso da configuraçaão 2)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=model_df2, x='PCA5', y='PCA6', hue='Cluster', palette='viridis', alpha=0.7)
plt.title('G5: STDDEV X RESULTANT')
plt.savefig('G5: PCA5XPCA6.png')
plt.close()

#Gráfico 6: Absolute Deviation X Resultant (info talvez relevante: uso da configuraçaão 3)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=model_df3, x='PCA7', y='PCA8', hue='Cluster', palette='viridis', alpha=0.7)
plt.title('G6: ABSOLDEV x RESULTANT (Cluster da Config 3)')
plt.savefig('G6: PCA7XPCA8.png')
plt.close()
    



