from numpy import load

data = load("model_3_GBC_purse_seines.pkl", allow_pickle=True)
lst = data.files
for item in lst:
    print(item)
    print(data[item])