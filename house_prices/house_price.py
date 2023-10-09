#invite people for the Kaggle party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf 

df_train = pd.read_csv("train.csv")
print(df_train.columns)