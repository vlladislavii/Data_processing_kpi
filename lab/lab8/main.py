import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 4. Вбудовані дані (І квартал 2015) ---
labels = [
    "М'ясо та птиця свіжі та заморожені", "М'ясо копчене, солоне та ковбасні вироби",
    "Консерви, готові продукти м'ясні", "Напівфабрикати м'ясні", "Риба і морепродукти харчові",
    "Консерви рибні", "Напівфабрикати рибні", "Сир сичужний та кисломолочний", "Масло вершкове",
    "Яйця", "Олії рослинні", "Маргарин", "Цукор", "Хлібобулочні вироби",
    "Кондитерські вироби борошняні", "Кондитерські вироби цукрові", "Борошно", "Крупи",
    "Макаронні вироби", "Свіжі овочі", "Картопля", "Фрукти та ягоди", "Перероблені овочі/фрукти",
    "Консерви овочеві", "Консерви фруктово-ягідні", "Горілчані вироби", "Слабоалкогольні напої",
    "Вина", "Коньяк", "Ігристі вина", "Пиво", "Безалкогольні напої", "Соки",
    "Води мінеральні", "Чай/кава/какао/прянощі", "Чай", "Кава", "Сіль", "Йодована сіль"
]
X_values = [
    541572.4, 362945.1, 56329.0, 25675.1, 189189.5, 77575.7, 14986.4, 200410.5, 76163.2, 558540.8,
    310499.3, 38416.2, 473388.0, 1130220.3, 407344.4, 321888.2, 437108.3, 520746.0, 294890.4, 1051920.1,
    351936.2, 821060.3, 64698.6, 123524.8, 23188.8, 2661929.4, 1078581.0, 1744243.6, 538598.1, 665445.2,
    12737452.1, 13144977.4, 4373190.0, 14491449.4, 78098.5, 26342.2, 34877.1, 167806.7, 24738.9
]
# Створюємо DataFrame та вхідну матрицю
df = pd.DataFrame(X_values, index=labels, columns=['І квартал 2015'])
print("Матриця вхідних даних:\n", df)

# --- 5. Стандартизація даних ---
scaler = StandardScaler()
X_norm = scaler.fit_transform(df.values)
df_norm = pd.DataFrame(X_norm, index=labels, columns=['Нормовані значення'])
print("\nМатриця нормованих даних:\n", df_norm)

# --- 5.1 Побудова кореляційної матриці ---
R_raw = np.corrcoef(X_norm, rowvar=False)
R = np.atleast_2d(R_raw)
print("\nКореляційна матриця R:\n", R)

# --- 6. Перевірка значущості відмінності від одиничної матриці ---
off = np.sum(R**2) - np.sum(np.diag(R)**2)
n = X_norm.shape[0]
d = n * off
k = X_norm.shape[1]
df_chi2 = k*(k-1)/2
chi2_crit = chi2.ppf(0.95, df_chi2) if df_chi2>0 else np.nan
print(f"\nСтатистика d = {d:.4f}, χ²_крит = {chi2_crit}")
print("PCA доцільний." if d > chi2_crit else "PCA недоцільний.")

# --- 7. Розрахунок проекцій на ГК (PCA через sklearn) ---
pca = PCA()
Scores = pca.fit_transform(X_norm)
Loadings = pca.components_.T
explained_variance = pca.explained_variance_
explained_ratio = pca.explained_variance_ratio_

# --- 8. Вивід матриць ---
print("\nМатриця навантажень (Loadings):\n",
      pd.DataFrame(Loadings, index=df.columns, columns=[f'PC{i+1}' for i in range(Loadings.shape[1])]))
print("\nМатриця рахунків (Scores):\n",
      pd.DataFrame(Scores, index=labels, columns=[f'PC{i+1}' for i in range(Scores.shape[1])]))
Errors = X_norm - pca.inverse_transform(Scores)
print("\nМатриця помилок (Errors):\n",
      pd.DataFrame(Errors, index=labels, columns=['Error']))

# --- 9. Аналіз результатів ---. Аналіз результатів ---
print("\nВласні значення (дисперсії ГК):", explained_variance)
print("\nЧастки поясненої дисперсії:", explained_ratio)

# Скрі-плот
plt.figure(figsize=(6,4))
plt.plot(np.arange(1, len(explained_variance)+1), explained_variance, marker='o')
plt.title('Скрі-плот')
plt.xlabel('Номер компоненти')
plt.ylabel('Дисперсія')
plt.grid(True)
plt.show()

# Бар-чарт навантажень на ПК1
plt.figure(figsize=(6,4))
plt.bar(np.arange(len(labels)), Loadings[:,0])
plt.title('Навантаження на ПК1')
plt.xlabel('Ознака')
plt.ylabel('Loadings')
plt.xticks(np.arange(len(labels)), labels, rotation=90)
plt.tight_layout()
plt.show()

# --- 10. Перевірка сум дисперсій ---
total_var_orig = np.var(X_norm, axis=0, ddof=1).sum()
total_var_pc = np.var(Scores, axis=0, ddof=1).sum()
print(f"\nСума дисперсій початкових ознак: {total_var_orig:.4f}")
print(f"Сума дисперсій проекцій: {total_var_pc:.4f}")

# --- 11. Визначення відносних часток та матриці коваріації ---
print("\nВідносні частки дисперсії:", explained_ratio)
Cov = np.cov(Scores, rowvar=False)
print("\nМатриця коваріації проекцій:\n", Cov)

# --- 12. Scatter-plot за першими двома ГК ---
if Scores.shape[1] >= 2:
    plt.figure(figsize=(6,6))
    plt.scatter(Scores[:,0], Scores[:,1])
    for i, lab in enumerate(labels):
        plt.text(Scores[i,0], Scores[i,1], lab, fontsize=8)
    plt.title('Scatter PC1 vs PC2')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("\nІнтерпретація:\nPC1 описує загальні обсяги продажів.\nPC2 описує відмінності між групами товарів і напоїв.")
else:
    plt.figure(figsize=(6,4))
    plt.hist(Scores[:,0], bins=10)
    plt.title('Гістограма ПК1')
    plt.xlabel('PC1')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.show()
