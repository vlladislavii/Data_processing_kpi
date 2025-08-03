# lab9_variant7.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# 1. Підготовка даних
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
X = np.array(X_values).reshape(-1, 1)

# 2. Кластеризація повним перебором об'єктів
from itertools import combinations

def all_partitions(objects):
    if len(objects) == 1:
        return [[objects]]
    first, rest = objects[0], objects[1:]
    partitions = []
    for smaller in all_partitions(rest):
        for i, subset in enumerate(smaller):
            partitions.append(smaller[:i] + [[first] + subset] + smaller[i+1:])
        partitions.append([[first]] + smaller)
    return partitions

def calculate_criterion(partition, dist_matrix):
    total = 0
    for cluster in partition:
        if len(cluster) > 1:
            for i, a in enumerate(cluster):
                for b in cluster[i+1:]:
                    total += dist_matrix[a, b]
    return total

def full_enumeration_clustering(X, max_n=8):
    n = X.shape[0]
    if n > max_n:
        print(f"Повний перебір неможливий для n={n}. Демонстрація на перших {max_n} об'єктах.")
        idx = list(range(max_n))
    else:
        idx = list(range(n))
    D = euclidean_distances(X[idx])
    parts = all_partitions(idx)
    best, best_val = None, np.inf
    for p in parts:
        val = calculate_criterion(p, D)
        if val < best_val:
            best, best_val = p, val
    print("Найкраще розбиття (демо):", best)
    print("Значення критерію:", best_val)
    return best

# 3. Кластеризація фіксованим радіусом

def fixed_radius_spherical_clustering(X, radius):
    unassigned = set(range(len(X)))
    clusters = []
    while unassigned:
        center_idx = unassigned.pop()
        center = X[center_idx]
        cluster = {center_idx}
        changed = True
        while changed:
            changed = False
            for j in list(unassigned):
                if np.linalg.norm(X[j] - center) <= radius:
                    cluster.add(j)
                    unassigned.remove(j)
                    changed = True
        clusters.append(sorted(cluster))
    print(f"Кластеризація фіксованим радіусом (r={radius:.2f}):", clusters)
    return clusters

# 4. Двоступенева сферична кластеризація з ядром

def two_stage_spherical_clustering(X, radius, density_threshold):
    D = euclidean_distances(X)
    cores = [i for i, row in enumerate(D) if np.sum(row <= radius) >= density_threshold]
    clusters = [[] for _ in cores]
    noise = []
    for i, x in enumerate(X):
        assigned = False
        for ci, core_idx in enumerate(cores):
            if np.linalg.norm(x - X[core_idx]) <= radius:
                clusters[ci].append(i)
                assigned = True
                break
        if not assigned:
            noise.append(i)
    print(f"Двоступенева сферична кластеризація (r={radius:.2f}, щільність>={density_threshold}):")
    for ci, cl in enumerate(clusters):
        print(f"  Кластер {ci+1} (ядро {cores[ci]}):", cl)
    print("Шумові точки:", noise)
    return clusters, noise

# 5. Інтегральна геометризація

def integral_geometrization_clustering(X, radius):
    clusters = []
    for i, x in enumerate(X):
        placed = False
        for cl in clusters:
            if np.linalg.norm(x - cl['center']) <= radius:
                cl['points'].append(i)
                placed = True
                break
        if not placed:
            clusters.append({'center': x, 'points': [i]})
    print(f"Інтегральна геометризація (r={radius:.2f}):")
    for i, cl in enumerate(clusters):
        print(f"  Кластер {i+1}, центр={cl['center'][0]:.2f}: точки {cl['points']}")
    return clusters

# 6. Кластеризація за середньою відстанню

def center_by_mean_distance_clustering(X, d_star):
    unclustered = set(range(len(X)))
    clusters = []
    while unclustered:
        curr = unclustered.pop()
        cluster = [curr]
        changed = True
        while changed:
            center = X[cluster].mean(axis=0)
            dists = {j: np.linalg.norm(X[j] - center) for j in unclustered}
            if not dists:
                break
            nearest, dist = min(dists.items(), key=lambda kv: kv[1])
            if dist <= d_star:
                cluster.append(nearest)
                unclustered.remove(nearest)
            else:
                changed = False
        clusters.append(sorted(cluster))
    print(f"Кластеризація за середньою відстанню (d*={d_star:.2f}):", clusters)
    return clusters

# 7. Виклик методів та графіки
if __name__ == "__main__":
    # Параметри кластеризації
    r1 = np.std(X_values) * 0.5
    r2 = np.std(X_values) * 0.7
    d_star = np.std(X_values) * 0.4

    # Запуск методів
    full_enumeration_clustering(X)
    clusters_fixed = fixed_radius_spherical_clustering(X, r1)
    clusters_two, noise = two_stage_spherical_clustering(X, r2, density_threshold=5)
    integral_geometrization_clustering(X, np.std(X_values)*0.6)
    center_by_mean_distance_clustering(X, d_star)

    # Функція для отримання ID кластерів для графіків
    def get_ids(clusters, noise=None):
        ids = np.full(len(X), -1)
        for cid, cl in enumerate(clusters):
            for idx in cl:
                ids[idx] = cid
        if noise:
            for idx in noise:
                ids[idx] = len(clusters)
        return ids

    # Графік 1: фіксований радіус
    ids_fixed = get_ids(clusters_fixed)
    plt.figure()
    plt.scatter(range(len(X_values)), X_values, c=ids_fixed)
    plt.title("Кластеризація фіксованим радіусом")
    plt.xlabel("Індекс продукту")
    plt.ylabel("Обсяг продажів (Q1)")
    plt.show()

    # Графік 2: двоступенева сферична
    ids_two = get_ids(clusters_two, noise)
    plt.figure()
    plt.scatter(range(len(X_values)), X_values, c=ids_two)
    plt.title("Двоступенева сферична кластеризація")
    plt.xlabel("Індекс продукту")
    plt.ylabel("Обсяг продажів (Q1)")
    plt.show()

    # Графік 3: середня відстань
    ids_mean = get_ids(center_by_mean_distance_clustering(X, d_star))
    plt.figure()
    plt.scatter(range(len(X_values)), X_values, c=ids_mean)
    plt.title("Кластеризація за середньою відстанню")
    plt.xlabel("Індекс продукту")
    plt.ylabel("Обсяг продажів (Q1)")
    plt.show()