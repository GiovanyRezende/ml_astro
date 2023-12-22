# Projeto Machine Learning
*Este projeto tem o objetivo de criar um método de Machine Learning que classifique objetos astronômicos usando [o banco de dados SQL criado anteriormente](https://github.com/GiovanyRezende/sql_db_com_python).* O algoritmo utilizado foi a Árvore de Decisão, um modelo de aprendizado supervisionado que utiliza um conjunto de regras em forma de árvore de busca (por isso o nome *Árvore de Decisão*). À seguir há um exemplo que faz analogia ao funcionamento do algoritmo:
![](https://www.hashtagtreinamentos.com/wp-content/uploads/2022/11/Arvore-de-Decisao-1.png)

*Fonte: [https://www.hashtagtreinamentos.com/wp-content/uploads/2022/11/Arvore-de-Decisao-1.png](https://www.hashtagtreinamentos.com/wp-content/uploads/2022/11/Arvore-de-Decisao-1.png)*

# Os dados
*Os dados são os cadastros no [banco de dados db_astro](https://github.com/GiovanyRezende/sql_db_com_python).* **Por ter sido necessário, houve o cadastro de mais uma informação em tb_galaxia, por isso, o arquivo do banco SQL utilizado para o algoritmo se chama db_astro(inicio) e db_astro é o banco final e atualizado.**

# Os dados de treino
*Os dados utilizados para treino e classificação são os dados de ```tb_corpo_celeste```, com exceção das colunas ```nome```, ```id``` e ```id_sistema```, sendo que esses atributos não são necessários para o Machine Learning.* Como o objetivo é classificar um objeto, há duas condições que devem ser seguidas para o ML:
- A coluna que se deseja prever ou classificar deve ser composta de números inteiros (o que, com a normalização dos dados do banco, isso já se cumpre);
- As classificações devem todas ter os mesmos atributos (o que já se cumpre dentro da tabela ```tb_corpo_celeste```).

Além disso, foram extraídas das colunas pré-existentes outras colunas para gerar mais dados de treino, mas isso foi feito utilizando o Pandas. 

# O código e o Machine Learning

## Importação das bibliotecas e conexão com o banco SQL
```
import sqlite3
import csv
import pandas as pd
import math
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

conn = sqlite3.connect('db_astro.db')
cursor = conn.cursor()
```

## A extração dos dados com consulta SQL
Primeiramente, se cria uma lista com os nomes dos atributos desejados:
```
#Para conferir as colunas de uma tabela, se usa o comando:
#data=cursor.execute('''SELECT * FROM {Tabela desejada}''')

colunas = []
data = cursor.execute('''SELECT * FROM tb_corpo_celeste''')
for column in data.description:
  if column[0] == "id":
    pass
  elif column[0] == "id_sistema":
    pass
  elif column[0] == "nome":
    pass
  else:
    colunas = colunas + [column[0]]
```
Esse código será terá seu objetivo cumprido mais para frente. Posteriormente, se realiza a consulta SQL e faz seu armazenamento em uma variável:
```
cursor.execute('''SELECT
               massa_kg,
               raio_medio_km,
               temperatura_k,
               id_classificacao
               FROM tb_corpo_celeste''')
query = cursor.fetchall()
cursor.execute("SELECT classificacao FROM tb_classificacao")
classif = cursor.fetchall()
classificacao = []
for x in classif:
  classificacao = classificacao + [x[0]]
```
A lista ```classificacao``` também terá um uso futuro.

## Conversão para dataframe no Pandas
Agora que a consulta já foi armazenada, se faz a criação de um arquivo CSV, assim como exemplificado no [projeto do banco](https://github.com/GiovanyRezende/sql_db_com_python):
```
a_csv = 'dados.csv'

with open(a_csv, 'w', newline='') as arquivo_csv:
    escritor_csv = csv.writer(arquivo_csv)
    escritor_csv.writerow(colunas)
    for x in query:
        escritor_csv.writerow(x)

df = pd.read_csv(a_csv)
```

## Criação de mais colunas
Uma vez que ```df``` está criado, podemos fazer sua manipulação para extrair mais informações interessantes. Considerando que o objetivo é classificar objetos astronômicos, também é importante saber o volume do astro, assim como a densidade, aceleração da gravidade e luminosidade **(ATENÇÃO! Com a ausência dos dados do albedo de cada astro, não é possível saber suas luminosidades. Ainda que estrelas podem ser consideradas como emissores ideiais, o mesmo não pode se dizer de outras classificações. Por isso, um valor de simulação foi criado para esse caso, sendo o produto do raio em metros ao quadrado e da temperatura elevada à quarta potência.)**.
```
G = 6.67e-11

df['volume_m3'] = 4*math.pi*((df['raio_medio_km']*1000)**3)/3
df['densidade_kg/m3'] = df['massa_kg']/df['volume_m3']
df['g'] = G*df['massa_kg']/((df['raio_medio_km']*1000)**2)
df['r_m2t4'] = ((df['raio_medio_km']*1000)**2)*(df['temperatura_k']**4)
```

# O Machine Learning
A acurácia do modelo varia entre 80% e 90%, sendo mais provável o segundo caso. Para extrair o melhor rendimento possível, o modelo é testado em até 10 tentativas ou o desejado:
```
for x in range(0,11):
  X_train, X_test, y_train, y_test = train_test_split(
      df.drop('id_classificacao', axis=1),
      df['id_classificacao'],
      test_size=0.2,
      random_state=42
  )
  model = DecisionTreeClassifier()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  acuracia = accuracy_score(y_test, y_pred)
  precisao = precision_score(y_test, y_pred, average = 'micro')
  sensibilidade = recall_score(y_test, y_pred, average = 'micro')
  f1 = f1_score(y_test, y_pred, average = 'micro')
  conf_matrix = confusion_matrix(y_test, y_pred)
  if acuracia >= 0.9:
    break
```
Não só a acurácia é medida, como também precisão, sensibilidade (recall) e o F1-score. As definições de cada pontuação são:
-**Acurácia**: é a razão entre as instâncias verdadeiras e todas as instâncias;
-**Precisão**: é a razão entre as instâncias verdadeiramente positivas e todas as instâncias ditas como positivas;
-**Sensibilidade**: é a razão entre os verdadeiros positivos e a soma dos verdadeiros positivos com os falsos negativos;
-**F1-score**: é uma média harmônica da Precisão e da Sensibilidade.

Extraindo esses valores e a matriz confusão, temos o seguinte resultado:
```
print(f'Acurácia: {acuracia}')
print(f'Precisão: {precisao}')
print(f'Sensibilidade: {sensibilidade}')
print(f'F1-score: {f1}\n')
print('Matriz de Confusão:')
print(conf_matrix,'\n')

>>>
Acurácia: 0.9
Precisão: 0.9
Sensibilidade: 0.9
F1-score: 0.9

Matriz de Confusão:
[[3 0 0 0 0]
 [0 2 0 0 0]
 [0 0 1 0 0]
 [0 0 0 2 1]
 [0 0 0 0 1]] 
```
A matriz de confusão mostra o quanto os dados de teste foram classificados corretamente. Se considerarmos a primeira linha, podemos ver que 3 instâncias forma classificadas corretamente como planetas e nenhuma instância foi classificada errada. Resumindo o seu funcionamento, podemos concluir que houve maior confusão do modelo com a classificação de satélites, havendo uma instância classificada errada como asteroide. 
