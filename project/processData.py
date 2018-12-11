import numpy as np
import pandas as pd

df = pd.read_csv('all/train.csv',sep=',', encoding='latin-1')
print(df)
arr = np.array(df)
print(arr.shape)
print(arr)
