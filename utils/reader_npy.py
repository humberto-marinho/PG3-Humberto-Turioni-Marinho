import numpy as np

# Especifique o caminho completo para o arquivo NPY que você deseja carregar
caminho_arquivo = 'C:\\Users\\user\\Desktop\\tcc2\\GNN-FakeNews\\data\\UPFD\\gossipcop\\raw\\graph_labels.npy'

# Carregue o array a partir do arquivo NPY
loaded_arr = np.load(caminho_arquivo)
print(loaded_arr.shape[0])
for i in range(500):
    print(loaded_arr[i])