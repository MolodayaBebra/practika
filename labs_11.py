import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('bodyPerformance.csv')

le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])

X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Матрица:")
print(confusion_mat)

# Plotting the error with increasing epochs
train_loss = model.loss_curve_

plt.plot(range(1, len(train_loss) + 1), train_loss)
plt.xlabel('Эпохи')
plt.ylabel('Ошибки')
plt.title('График ошибки при обучении с ростом числа эпох')
plt.show()
