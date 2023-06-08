import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("bodyPerformance.csv")

# 1. Изучить распределение целевых классов и нескольких категориальных признаков
sns.countplot(x='class', data=df)
plt.show()

sns.countplot(x='gender', data=df)
plt.show()

# 2. Нарисовать распределения нескольких числовых признаков
sns.histplot(data=df, x='height_cm')
plt.show()

sns.histplot(data=df, x='weight_kg')
plt.show()

# 3. Произвести нормализацию нескольких числовых признаков
numeric_features = df[['height_cm', 'weight_kg']]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numeric_features)
normalized_df = pd.DataFrame(normalized_data, columns=numeric_features.columns)
print(normalized_df.head())

# 4. Посмотреть и визуализировать корреляцию признаков
numeric_columns = ['height_cm', 'weight_kg']
categorical_columns = ['gender']

# Преобразование категориальных признаков в числовые индикаторы
encoded_df = pd.get_dummies(df, columns=categorical_columns)

corr = encoded_df[numeric_columns].corr()
sns.heatmap(corr, cmap="crest")
plt.show()
