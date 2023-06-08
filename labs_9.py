import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('bodyPerformance.csv')

le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])

X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
lr_model = LogisticRegression(max_iter=10000)

model = VotingClassifier(
    estimators=[('dt', dt_model), ('knn', knn_model), ('lr', lr_model)],
    voting='hard'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Матрица:")
print(confusion_mat)

param_grid = {
    'dt__criterion': ['gini', 'entropy'],
    'dt__max_depth': [None, 5, 10],
    'knn__n_neighbors': [3, 5, 7],
    'lr__C': [0.01, 0.1, 1, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred_new = best_model.predict(X_test)

accuracy_new = accuracy_score(y_test, y_pred_new)
precision_new = precision_score(y_test, y_pred_new, average='weighted')

print("\nМетрики с оптимальными параметрами:")
print("Accuracy:", accuracy_new)
print("Precision:", precision_new)
