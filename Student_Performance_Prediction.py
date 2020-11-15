# Importing Libraries
import numpy as np 
import pandas as pd 
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing The Dataset
dataset = pd.read_csv('./Data/student-mat.csv', sep=';')
X = dataset.iloc[:, 0:32]
X2 = dataset.iloc[:, 0:32]
y = dataset.iloc[:, 32]








































