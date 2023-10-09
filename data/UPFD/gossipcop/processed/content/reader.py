import torch

# Especifique o caminho para o arquivo .pt que você deseja ler
path_to_file = 'data\\UPFD\\gossipcop\\processed\\content\\val.pt'

# Use a função torch.load() para ler o arquivo .pt
data = torch.load(path_to_file)

# Agora você pode trabalhar com os dados carregados
# Por exemplo, você pode imprimir os dados ou acessar suas propriedades
print(data)