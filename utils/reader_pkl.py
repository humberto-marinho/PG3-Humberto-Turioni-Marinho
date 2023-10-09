import pickle
file_path = "C:\\Users\\user\\Desktop\\tcc2\\GNN-FakeNews\\data\\gos_id_twitter_mapping.pkl"
with open(file_path, 'rb') as file:
    object = pickle.load(file)
print(str(object)[0:5000])