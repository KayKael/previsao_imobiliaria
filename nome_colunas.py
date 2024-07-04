import pandas as pd

# Carregar os dados
data_pt = pd.read_csv('C:\\Users\\kaell\\Desktop\\projeto_imobiliario\\prc_hpi_q_linear.csv')

# Exibir os nomes das colunas
print(data_pt.columns)

# Mostrar todos os valores únicos da coluna "geo"
valores_unicos_geo = data_pt['geo'].unique()
print(valores_unicos_geo)
