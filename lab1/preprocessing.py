#1. Загрузить данные с Kaggle – выбрать датасет (например, Титаник)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

print("Данные загружены, первые 5 строк: ")
print(df.head())
