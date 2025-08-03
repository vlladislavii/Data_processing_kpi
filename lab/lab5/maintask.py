import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist, euclidean
import warnings

warnings.filterwarnings("ignore")

# ==================================================
# 1. ПІДГОТОВКА ДАНИХ ТА ПЕРВИННА ВІЗУАЛІЗАЦІЯ
# ==================================================
data = [
    ("М'ясо та птиця свіжі та заморожені", 541572.4),
    ("М'ясо копчене, солоне та ковбасні вироби", 362945.1),
    ("Консерви, готові продукти м'ясні", 56329.0),
    ("у тому числі напівфабрикати м'ясні", 25675.1),
    ("Риба і морепродукти харчові", 189189.5),
    ("Консерви, готові продукти рибні", 77575.7),
    ("у тому числі напівфабрикати рибні", 14986.4),
    ("Сир сичужний, плавлений та кисломолочний", 200410.5),
    ("Масло вершкове", 76163.2),
    ("Яйця", 558540.8),
    ("Олії рослинні", 310499.3),
    ("Маргарин", 38416.2),
    ("Цукор", 473388.0),
    ("Вироби хлібобулочні (крім кондитерських)", 1130220.3),
    ("Вироби борошняні кондитерські", 407344.4),
    ("Вироби цукрові кондитерські (включаючи морозиво)", 321888.2),
    ("Борошно", 437108.3),
    ("Крупи", 520746.0),
    ("Вироби макаронні", 294890.4),
    ("Свіжі овочі", 1051920.1),
    ("у тому числі картопля", 351936.2),
    ("Свіжі плоди, ягоди, виноград, горіхи", 821060.3),
    ("Овочі та фрукти перероблені", 64698.6),
    ("Консерви овочеві", 123524.8),
    ("Консерви фруктово-ягідні", 23188.8),
    ("Горілка та вироби лікеро-горілчані", 2661929.4),
    ("Напої слабоалкогольні", 1078581.0),
    ("Вина", 1744243.6),
    ("Коньяк", 538598.1),
    ("Вина ігристі (шампанське)", 665445.2),
    ("Пиво", 12737452.1),
    ("Напої безалкогольні", 13144977.4),
    ("у тому числі соки", 4373190.0),
    ("Води мінеральні", 14491449.4),
    ("Чай, кава, какао та прянощі", 78098.5),
    ("чай", 26342.2),
    ("кава", 34877.1),
    ("Сіль", 167806.7),
    ("у тому числі йодована кухонна сіль", 24738.9)
]

df = pd.DataFrame(data, columns=["Продукт", "Продажі"])
df["Індекс"] = np.arange(1, len(df) + 1)

# Формуємо 2D-матрицю ознак: [Індекс, Продажі]
X = df[["Індекс", "Продажі"]].values

# Первинна візуалізація (розсіювання)
plt.figure(figsize=(10, 6))
plt.scatter(df["Індекс"], df["Продажі"], color='blue')
plt.title("Експериментальні дані: Індекс продукту vs. Продажі")
plt.xlabel("Індекс продукту")
plt.ylabel("Продажі (І квартал 2015)")
plt.grid(True)
plt.show()

# ==================================================
# 2. НАЛАШТУВАННЯ ПАРАМЕТРІВ КЛАСТЕРИЗАЦІЇ
# ==================================================
metrics = {
    "Евклідова": "euclidean",
    "Махаланобіса": "mahalanobis",
    "Чебишева": "chebyshev"
}

linkage_methods = {
    "Complete": "complete",  # дальший сусід
    "Average": "average",  # середній зв’язок
    "Centroid": "centroid"  # центроїдний метод
}

# Для Махаланобіса обчислюємо обернену коваріаційну матрицю
cov_matrix = np.cov(X.T)
inv_cov_matrix = np.linalg.inv(cov_matrix)

# К-сть кластерів (завдання вимагає 3)
num_clusters = 3

# Список для збереження даних, щоб потім вивести кофенет. коефіцієнти
all_results = []

# ==================================================
# 3. ПОБУДОВА 9 ПАР ГРАФІКІВ (ДЕНДРОГРАМА + SCATTER)
#    ДЛЯ КОЖНОЇ КОМБІНАЦІЇ (3×3)
# ==================================================
for met_name, met in metrics.items():
    # Обчислюємо відстані d
    if met == "mahalanobis":
        d = pdist(X, metric=met, VI=inv_cov_matrix)
    else:
        d = pdist(X, metric=met)

    for link_name, link_method in linkage_methods.items():
        # Отримуємо ієрархічне дерево
        Z = linkage(d, method=link_method)
        # Кофенетичний коефіцієнт
        coph_corr, _ = cophenet(Z, d)
        all_results.append({
            "Метрика": met_name,
            "Метод зв’язування": link_name,
            "Кофенетичний": coph_corr,
            "Z": Z,
            "d": d
        })

        # Визначення порогового значення для кольорової дендрограми
        cutoff = np.median([Z[-2, 2], Z[-1, 2]])

        # -------------------------------
        # (a) Дендрограма
        # -------------------------------
        plt.figure(figsize=(10, 7))
        dendrogram(Z,
                   labels=df["Продукт"].values,
                   leaf_rotation=90,
                   color_threshold=cutoff)
        plt.title(f"Дендрограма: {met_name} - {link_name}\n(Кол. поріг={cutoff:.2f})")
        plt.xlabel("Продукти")
        plt.ylabel("Відстань")
        plt.tight_layout()
        plt.show()

        # -------------------------------
        # (b) Кластеризація і scatter
        # -------------------------------
        clusters = fcluster(Z, num_clusters, criterion='maxclust')
        df["Кластер"] = clusters

        plt.figure(figsize=(10, 6))
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        for cluster_id in sorted(np.unique(clusters)):
            subset = df[df["Кластер"] == cluster_id]
            plt.scatter(subset["Індекс"], subset["Продажі"],
                        color=colors[cluster_id % len(colors)],
                        label=f"Кластер {cluster_id}")
        plt.title(f"Розподіл на кластери: {met_name} - {link_name}")
        plt.xlabel("Індекс продукту")
        plt.ylabel("Продажі (І квартал 2015)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ==================================================
# 4. АНАЛІЗ РЕЗУЛЬТАТІВ У КОНСОЛІ
# ==================================================
df_results = pd.DataFrame(all_results)
print("\n=== ТАБЛИЦЯ КОФЕНЕТИЧНИХ КОЕФІЦІЄНТІВ (УСІ КОМБІНАЦІЇ) ===")
print(df_results[["Метрика", "Метод зв’язування", "Кофенетичний"]])

best_row = df_results.loc[df_results["Кофенетичний"].idxmax()]
worst_row = df_results.loc[df_results["Кофенетичний"].idxmin()]

print("\nНАЙКРАЩИЙ СПОСІБ КЛАСТЕРИЗАЦІЇ:")
print(best_row[["Метрика", "Метод зв’язування", "Кофенетичний"]])
print("\nНАЙГІРШИЙ СПОСІБ КЛАСТЕРИЗАЦІЇ:")
print(worst_row[["Метрика", "Метод зв’язування", "Кофенетичний"]])

# ==================================================
# 5. ДОДАТКОВИЙ АНАЛІЗ ДЛЯ НАЙКРАЩОГО МЕТОДУ
# ==================================================
print("\n=== АНАЛІЗ НАЙКРАЩОГО МЕТОДУ ===")
Z_best = best_row["Z"]
d_best = best_row["d"]
met_best = best_row["Метрика"]
link_best = best_row["Метод зв’язування"]

# Порогове значення (медіана останніх двох злиттів)
cutoff_best = np.median([Z_best[-2, 2], Z_best[-1, 2]])

# (1) Дендрограма з порогом
plt.figure(figsize=(10, 7))
dendrogram(Z_best,
           labels=df["Продукт"].values,
           leaf_rotation=90,
           color_threshold=cutoff_best)
plt.title(f"Дендрограма для найкращого методу: {met_best} - {link_best}\n(Кол.поріг={cutoff_best:.2f})")
plt.xlabel("Продукти")
plt.ylabel("Відстань")
plt.tight_layout()
plt.show()

# (2) Формуємо кластери (3)
best_clusters = fcluster(Z_best, num_clusters, criterion='maxclust')
df["Кластер"] = best_clusters

print("\nРозподіл продуктів по кластерах (найкращий метод):")
print(df[["Продукт", "Продажі", "Кластер"]])

# (3) Центри та дисперсія
cluster_centers = df.groupby("Кластер")[["Індекс", "Продажі"]].mean()
cluster_disp = df.groupby("Кластер")[["Індекс", "Продажі"]].std()
print("\nЦентри кластерів (середні значення):")
print(cluster_centers)
print("\nВнутрішньокластерна дисперсія (std):")
print(cluster_disp)


# (4) Відстані від кожного об’єкта до центру кластера
def dist_to_center(row):
    c = cluster_centers.loc[row["Кластер"]]
    return euclidean((row["Індекс"], row["Продажі"]),
                     (c["Індекс"], c["Продажі"]))


df["Відстань_до_центру"] = df.apply(dist_to_center, axis=1)
print("\nВідстані кожного елемента до центру свого кластера:")
print(df[["Продукт", "Кластер", "Відстань_до_центру"]])

# (5) Відстані між центрами кластерів
from itertools import combinations

center_ids = cluster_centers.index
dist_dict = {}
for (i, j) in combinations(center_ids, 2):
    ci = cluster_centers.loc[i]
    cj = cluster_centers.loc[j]
    d_cij = euclidean((ci["Індекс"], ci["Продажі"]),
                      (cj["Індекс"], cj["Продажі"]))
    dist_dict[f"({i},{j})"] = d_cij

print("\nВідстані між центрами кластерів:")
for k, v in dist_dict.items():
    print(f"{k}: {v:.2f}")

# (6) Фінальне розсіювання найкращого методу (з центрами)
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue', 'purple', 'orange']
for cl in sorted(np.unique(best_clusters)):
    subset = df[df["Кластер"] == cl]
    plt.scatter(subset["Індекс"], subset["Продажі"],
                color=colors[cl % len(colors)],
                label=f"Кластер {cl}")

# Позначимо центри
for cl in cluster_centers.index:
    cx, cy = cluster_centers.loc[cl]["Індекс"], cluster_centers.loc[cl]["Продажі"]
    plt.plot(cx, cy, 'ko', markersize=10, markeredgecolor='k')

plt.title(f"Найкращий метод: {met_best} - {link_best} (3 кластери)")
plt.xlabel("Індекс продукту")
plt.ylabel("Продажі (І квартал 2015)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
