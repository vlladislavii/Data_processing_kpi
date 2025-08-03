import pandas as pd
from itertools import chain, combinations
import matplotlib.pyplot as plt

# 1. Завантаження CSV у DataFrame
def load_dataset(filename="shopping_data.csv"):
    """
    Завантажує CSV-файл у формат pandas DataFrame
    """
    return pd.read_csv(filename)

# 2. Перетворення DataFrame з 0/1 у список транзакцій
def to_transactions(df):
    """
    Конвертує кожен рядок (з 0/1) у список товарів,
    де row[col] == 1.
    """
    transactions = []
    for _, row in df.iterrows():
        items = []
        for col in df.columns:
            if row[col] == 1:
                items.append(col)
        transactions.append(items)
    return transactions

# 3. Побудова вертикального представлення (tid_list)
def build_tid_list(transactions):
    """
    Повертає словник: товар -> множина індексів транзакцій (tid),
    де цей товар зустрічається.
    """
    tid_list = {}
    for tid, transaction in enumerate(transactions):
        for item in transaction:
            if item not in tid_list:
                tid_list[item] = set()
            tid_list[item].add(tid)
    return tid_list

# 4. Рекурсивна функція ECLAT
def eclat_recursive(items, tid_list, prefix, idx, min_support_count, frequent_itemsets):
    """
    items: відсортований список унікальних товарів
    tid_list: словник {item: set(tids)}
    prefix: поточний накопичений набір (список товарів)
    idx: індекс початку ітерації
    min_support_count: абсолютне значення мінімальної підтримки
    frequent_itemsets: словник для зберігання результатів
                       {tuple(sorted_items): count}
    """
    for i in range(idx, len(items)):
        item = items[i]
        new_prefix = prefix + [item]
        new_tid_set = tid_list[item]

        if len(new_tid_set) >= min_support_count:
            frequent_itemsets[tuple(sorted(new_prefix))] = len(new_tid_set)

            for j in range(i + 1, len(items)):
                next_item = items[j]
                combined_tid_set = new_tid_set & tid_list[next_item]
                if len(combined_tid_set) >= min_support_count:
                    tid_list[next_item] = combined_tid_set

            eclat_recursive(items, tid_list, new_prefix, i + 1, min_support_count, frequent_itemsets)

# 5. Основна функція ECLAT
def eclat(transactions, min_support=0.01):
    """
    Повертає словник частих наборів (keys) та їх підтримку
    (кількість транзакцій) (values).
    min_support = 0.01 означає 1%.
    """
    n_transactions = len(transactions)
    min_support_count = int(min_support * n_transactions) if min_support < 1 else int(min_support)

    tid_list = build_tid_list(transactions)
    items_sorted = sorted(tid_list.keys(), key=lambda x: len(tid_list[x]), reverse=True)

    frequent_itemsets = {}
    eclat_recursive(items_sorted, tid_list, [], 0, min_support_count, frequent_itemsets)

    return frequent_itemsets, n_transactions

# 6. Форматуємо дані частих наборів у відсотки
def format_eclat_results(frequent_itemsets, total_transactions):
    """
    Перетворює кількість транзакцій у відсоток підтримки
    та формує словник:
    {
      'milk | spaghetti': '2.0000%',
      'bread | butter | chicken': '1.0000%',
      ...
    }
    """
    results = {}
    for itemset, count in frequent_itemsets.items():
        itemset_str = " | ".join(itemset)
        support_percent = (count / total_transactions) * 100
        results[itemset_str] = f"{support_percent:.4f}%"
    return results

# 7. Побудова правил: (A -> B) для кожного частого набору
def compute_association_rules(frequent_itemsets, total_transactions):
    """
    Генерує всі можливі партиції набору S на (A,B),
    де A ∪ B = S і A ∩ B = ∅.
    Обчислює:
      - support = sup(A ∪ B)
      - confidence = sup(A ∪ B) / sup(A)
      - lift = sup(A ∪ B) / (sup(A)*sup(B))
      - conviction = (1 - sup(B)) / (1 - confidence)
    Повертає список словників.
    """
    rules = []
    freq_map = {k: v for k, v in frequent_itemsets.items()}

    for itemset, sup_count in freq_map.items():
        if len(itemset) < 2:
            continue

        from itertools import chain, combinations
        all_subsets = chain.from_iterable(
            combinations(itemset, r) for r in range(1, len(itemset))
        )

        for subset in all_subsets:
            A = tuple(sorted(subset))
            B = tuple(sorted(set(itemset) - set(A)))
            if not A or not B:
                continue

            supAUB_count = sup_count
            supAUB = supAUB_count / total_transactions

            if A not in freq_map:
                continue
            supA = freq_map[A] / total_transactions

            if B not in freq_map:
                continue
            supB = freq_map[B] / total_transactions

            if supA == 0:
                continue
            confidence = supAUB / supA

            lift = 0.0
            if (supA * supB) != 0:
                lift = supAUB / (supA * supB)

            conviction = 0.0
            if confidence < 1.0:
                conviction = (1 - supB) / (1 - confidence)

            rule_info = {
                'antecedent': A,
                'consequent': B,
                'support': round(supAUB, 4),
                'confidence': round(confidence, 4),
                'lift': round(lift, 4),
                'conviction': round(conviction, 4)
            }
            rules.append(rule_info)
    return rules

# 8. Вивід кількох правил (за певною метрикою)
def print_top_rules(rules, metric='lift', top_n=10):
    """
    Виводить ТОП-N правил за заданою метрикою ('confidence', 'lift' або 'conviction').
    """
    rules_sorted = sorted(rules, key=lambda r: r[metric], reverse=True)

    print(f"\nТОП-{top_n} правил (за {metric}):\n")
    for r in rules_sorted[:top_n]:
        A_str = ", ".join(r['antecedent'])
        B_str = ", ".join(r['consequent'])
        print(f"Правило: {{{A_str}}} -> {{{B_str}}} | "
              f"support={r['support']}, "
              f"confidence={r['confidence']}, "
              f"lift={r['lift']}, "
              f"conviction={r['conviction']}")

# 9. Візуалізація: Scatter Plot
def visualize_rules_scatter(rules):
    """
    Будує Scatter Plot:
    - X: support
    - Y: confidence
    - Розмір точки: lift
    - Колір точки: conviction
    """
    if not rules:
        print("\nНемає правил для відображення (список rules порожній).")
        return

    df_rules = pd.DataFrame(rules)
    if df_rules.empty:
        print("\nНемає правил для відображення (DataFrame порожній).")
        return

    x = df_rules['support'].astype(float)
    y = df_rules['confidence'].astype(float)
    sizes = df_rules['lift'].astype(float) * 100
    colors = df_rules['conviction'].astype(float)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, s=sizes, c=colors, cmap='viridis', alpha=0.6, edgecolors='k')
    plt.xlabel('Підтримка (support)')
    plt.ylabel('Достовірність (confidence)')
    plt.title('Асоціативні правила (ECLAT): Support vs Confidence')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Переконливість (conviction)')
    plt.show()

# 10. Візуалізація: Горизонтальний Bar Chart (ТОП-N)
def visualize_rules_bar(rules, metric='lift', top_n=10):
    """
    Будуємо горизонтальний Bar Chart для ТОП-N правил за обраною метрикою.
    """
    if not rules:
        print("\nНемає правил для відображення (список rules порожній).")
        return

    df_rules = pd.DataFrame(rules)
    if df_rules.empty:
        print("\nНемає правил для відображення (DataFrame порожній).")
        return

    df_sorted = df_rules.sort_values(by=metric, ascending=False)
    top_rules = df_sorted.head(top_n)

    labels = []
    values = []
    for _, row in top_rules.iterrows():
        A_str = ", ".join(row['antecedent'])
        B_str = ", ".join(row['consequent'])
        labels.append(f"{A_str} -> {B_str}")
        values.append(row[metric])

    plt.figure(figsize=(8, 6))
    y_pos = range(len(values))
    plt.barh(y_pos, values, color='skyblue')
    plt.gca().invert_yaxis()
    plt.yticks(y_pos, labels, fontsize=8)
    plt.xlabel(metric.capitalize())
    plt.title(f"Топ-{top_n} правил (за {metric})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Завантажимо датасет
    df = load_dataset("shopping_data.csv")
    transactions = to_transactions(df)

    # ECLAT з мінімальною підтримкою, наприклад 1%
    frequent_itemsets, total_trans = eclat(transactions, min_support=0.01)

    # Формуємо та виводимо список частих наборів (у відсотках)
    formatted_results = format_eclat_results(frequent_itemsets, total_trans)
    sorted_frequent = sorted(
        formatted_results.items(),
        key=lambda x: float(x[1].replace('%', '')),
        reverse=True
    )
    print("=== Часті набори (ECLAT) ===")
    for k, v in sorted_frequent:
        print(f"'{k}': '{v}',")

    # Генеруємо правила та обчислюємо conf, lift, conv
    rules = compute_association_rules(frequent_itemsets, total_trans)
    print(f"\nЗагалом згенеровано правил: {len(rules)}")

    # Показуємо ТОП-10 за lift
    print_top_rules(rules, metric='lift', top_n=10)

    # Побудова діаграм
    visualize_rules_scatter(rules)
    visualize_rules_bar(rules, metric='lift', top_n=10)
