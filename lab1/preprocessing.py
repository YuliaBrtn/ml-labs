#1. Загрузить данные с Kaggle – выбрать датасет (например, Титаник)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

print("Данные загружены, первые 5 строк: ")
print(df.head())


#2. Вывести с помощью python данные из датасета на экран
print("\nИнформация о датасете:")
df.info()

print("\nСтатистическое описание числовых столбцов:")
print(df.describe())

print("\nНазвания всех столбцов:")
print(df.columns.tolist())


#3. Получить количество пропущенных значений для каждого столбца в датасетах
missing = df.isnull().sum()
print("\nПропущенные значения по столбцам:")
print(missing[missing > 0])