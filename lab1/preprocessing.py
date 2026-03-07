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


#4. Заполнить пропущенные значения в датасете модой/медианой/средним значением и показать, что они действительно были заполнены
print("\nУникальные значения в Sleep Disorder до заполнения:")
print(df['Sleep Disorder'].value_counts(dropna=False))

mode_sleep = df['Sleep Disorder'].mode()[0]
df['Sleep Disorder'] = df['Sleep Disorder'].fillna(mode_sleep)   # исправлено!

print(f"\nСтолбец Sleep Disorder заполнен модой: '{mode_sleep}'")

print("\nПропуски после заполнения:")
print(df.isnull().sum())


#5. Провести нормализацию данных
numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep',
                'Physical Activity Level', 'Stress Level',
                'Heart Rate', 'Daily Steps']

print("\nЧисловые столбцы для нормализации:", numeric_cols)

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nПервые 5 строк после нормализации (только числовые столбцы):")
print(df[numeric_cols].head())