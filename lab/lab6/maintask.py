# Імпортуємо необхідні бібліотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # <-- додано PCA

# 2. Завантаження даних вручну (варіант 7)
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

# Перетворення у масив значень
labels = [item[0] for item in data]
values = np.array([item[1] for item in data]).reshape(-1, 1)

# Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(values)

# Метод Ліктя
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Метод Силуету
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=0)
    score = silhouette_score(X_scaled, kmeans.fit_predict(X_scaled))
    silhouette_scores.append(score)

# Побудова графіків (Ліктя та Силуету)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Метод Ліктя")
plt.xlabel("Кількість кластерів")
plt.ylabel("WCSS")

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
plt.title("Метод Силуету")
plt.xlabel("Кількість кластерів")
plt.ylabel("Силуетний коефіцієнт")
plt.tight_layout()
plt.show()

# --- Реалізація кластеризації методом Мінковського ---
class KMeansMinkowski:
    def __init__(self, n_clusters=3, max_iter=300, p=4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.p = p

    def fit(self, X):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[indices]
        for _ in range(self.max_iter):
            distances = pairwise_distances(X, self.centroids, metric='minkowski', p=self.p)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        self.labels_ = labels
        return self

# --- Кластеризація ---
model = KMeansMinkowski(n_clusters=3, p=4)
model.fit(X_scaled)

# --- Q1 ---
Q1 = sum(
    np.linalg.norm(X_scaled[i] - model.centroids[model.labels_[i]], ord=4)
    for i in range(len(X_scaled))
)

# --- Друк результатів ---
print("\n🧠 Обране значення кластерів: k = 3 (згідно графіків)")
print("\n📌 Розраховані центри кластерів (нормалізовані):")
for i, c in enumerate(model.centroids):
    print(f"   - Кластер {i}: Центроїд = {c[0]:.4f}")
print(f"\n📐 Функціонал якості кластеризації Q1 = {Q1:.4f}")

print("\n📋 Розподіл об'єктів по кластерах:")
for cluster_id in range(3):
    print(f"\n🔹 Кластер {cluster_id}:")
    for i, label in enumerate(model.labels_):
        if label == cluster_id:
            print(f"   - {labels[i]}")

# --- ВІЗУАЛІЗАЦІЯ ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=model.labels_, cmap='viridis', s=100)
plt.title("Двовимірна візуалізація кластерів (PCA + Мінковський, p=4)")
plt.xlabel("Головна компонента 1")
plt.ylabel("Головна компонента 2")
plt.grid(True)
plt.show()
