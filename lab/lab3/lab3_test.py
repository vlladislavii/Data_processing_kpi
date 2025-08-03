from ftplib import print_line
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Крок 1: Сформувати набір даних для обробки та аналізу
np.random.seed(42)
data = pd.DataFrame({
    'Population': np.random.randint(50000, 1000000, 100),  # Населення міста
    'Area': np.random.randint(50, 500, 100),  # Площа міста (км²)
    'Average_Income': np.random.randint(20000, 100000, 100),  # Середній дохід (USD)
    'Has_University': np.random.choice([0, 1], 100),  # 1 - є університет, 0 - немає
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),  # Регіон
    'Economy': np.random.choice(['Developed', 'Developing', 'Underdeveloped'], 100),  # Статус економіки
    'Unemployment_Rate': np.random.uniform(2, 20, 100),  # Рівень безробіття (%)
    'Climate': np.random.choice(['Tropical', 'Temperate', 'Arctic'], 100),  # Тип клімату
    'Education_Budget': np.random.randint(1000000, 50000000, 100),  # Бюджет на освіту (USD)
    'Population_Density': np.random.randint(100, 20000, 100)  # Щільність населення (осіб/км²)
})

# Додавання нового індексу "Urbanization" для міст
data['Urbanization'] = np.random.choice(['Urban', 'Suburban', 'Rural'], 100)  # Статус урбанізації

# Перетворення категоріальних змінних у числові (One-Hot Encoding)
data = pd.get_dummies(data, drop_first=True)

# Крок 2: Розробити забезпечення для обробки набору даних з метою побудови дерева вирішальних правил
X = data.drop(columns=['Has_University'])
y = data['Has_University']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Побудова дерева рішень (алгоритм CART)
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Візуалізація дерева рішень
plt.figure(figsize=(25, 23))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['No University', 'Has University'], rounded=True, proportion=True)
plt.title("Decision Tree for University Presence in Cities")
plt.show()

# Крок 3: Використати побудоване дерево для прийняття рішень на конкретному прикладі
new_city = pd.DataFrame({
    'Population': [350000],  # Населення міста
    'Area': [150],  # Площа міста
    'Average_Income': [35000],  # Середній дохід (USD)
    'Unemployment_Rate': [7.5],  # Рівень безробіття (%)
    'Education_Budget': [12000000],  # Бюджет на освіту (USD)
    'Population_Density': [2333],  # Щільність населення (осіб/км²)
    # One-Hot Encoding для категоріальних змінних
    'Region_South': [0],  # Південний регіон (One-Hot Encoding)
    'Economy_Developing': [1],  # Розвиваюча економіка (One-Hot Encoding)
    'Climate_Temperate': [1],  # Помірний клімат (One-Hot Encoding)
    'Urbanization_Urban': [1],  # Урбанізоване місто (One-Hot Encoding)
    # Відсутні One-Hot закодовані категорії (встановлюємо все в 0, якщо цього немає в цьому місті)
    'Region_North': [0],
    'Region_West': [0],
    'Economy_Underdeveloped': [0],
    'Climate_Tropical': [0],
    'Urbanization_Suburban': [0]
})

# Переконаємося, що нове місто має ті самі стовпці, що й навчальні дані (такий самий набір ознак)
new_city = new_city[X.columns]

# Прогнозування для нового міста
prediction = dt_model.predict(new_city)
prediction_proba = dt_model.predict_proba(new_city)

# Виведення прогнозу та ймовірності
if prediction == 1:
    print("Місто, ймовірно, має університет.\n")
else:
    print("Місто, ймовірно, не має університету.\n")

# Виведення ймовірності кожного класу
print(f"Ймовірність наявності університету: {prediction_proba[0][1]:.2f}")
print(f"Ймовірність відсутності університету: {prediction_proba[0][0]:.2f}")

# Крок 4: Провести відсікання зайвих гілок отриманого дерева рішень
dt_model_pruned = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
dt_model_pruned.fit(X_train, y_train)

# Візуалізація усіченого дерева рішень
plt.figure(figsize=(14, 10))
plot_tree(dt_model_pruned, filled=True, feature_names=X.columns, class_names=['No University', 'Has University'], rounded=True, proportion=True)
plt.title("Pruned Decision Tree for University Presence in Cities")
plt.show()

# Крок 5: Обчислити точності класифікації для повного та усіченого дерев
y_pred = dt_model.predict(X_test)
y_pred_pruned = dt_model_pruned.predict(X_test)
print_line("")

# Точність, Precision та Recall для повного дерева рішень
print("Точність для повного дерева:")
print(f"Точність: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")

# Точність, Precision та Recall для усіченого дерева рішень
print("\nТочність для усіченого дерева:")
print(f"Точність: {accuracy_score(y_test, y_pred_pruned):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_pruned):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_pruned):.2f}")
