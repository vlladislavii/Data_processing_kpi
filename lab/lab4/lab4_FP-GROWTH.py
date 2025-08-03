import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 1. Завантаження CSV та формування транзакцій
def load_dataset(filename="shopping_data.csv"):
    """
    Зчитує CSV-файл із 0/1 і повертає DataFrame.
    Кожен рядок = транзакція,
    кожна колонка = товар (1 - куплено, 0 - ні).
    """
    return pd.read_csv(filename)

def df_to_transactions(df):
    """
    Конвертує DataFrame (0/1) у список транзакцій:
    [
      ['Milk','Bread'],
      ['Beer','Chips','Salt'],
      ...
    ]
    """
    transactions = []
    for _, row in df.iterrows():
        items = []
        for col in df.columns:
            if row[col] == 1:
                items.append(col)
        transactions.append(items)
    return transactions

def encode_transactions(transactions):
    """
    Використовує TransactionEncoder для
    перетворення списку списків у DataFrame (True/False).
    """
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    return df_encoded

# 2. Запуск FP-GROWTH
def run_fpgrowth(df_encoded, min_support=0.03):
    """
    Запускає fpgrowth з бібліотеки mlxtend.
    Повертає DataFrame із кол.:
      - support
      - itemsets (список товарів)
    """
    frequent_itemsets = fpgrowth(
        df_encoded,
        min_support=min_support,
        use_colnames=True
    )
    return frequent_itemsets

# 3. Генерація правил асоціації
def generate_rules(frequent_itemsets, min_confidence=0.4):
    """
    Використовує association_rules() для генерації правил:
    повертає DataFrame із:
      - antecedents
      - consequents
      - support (A ∪ B)
      - confidence
      - lift
      - conviction (якщо не додається автоматично)
    """
    from mlxtend.frequent_patterns import association_rules
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
    if 'conviction' not in rules.columns:
        # conviction = (1 - sup(B)) / (1 - confidence)
        # де sup(B) = 'consequent support'
        rules['conviction'] = ((1 - rules['consequent support'])
                               / (1 - rules['confidence']))
    return rules

# 4. Форматований вивід у стилі "items" + "ordered_statistics"
def format_rules_for_display(rules):
    """
    Створює DataFrame із трьома колонками:
      'items': об'єднання base + add
      'ordered_statistics': [{'items_base':[...], 'items_add':[...], 'confidence':..., 'lift':..., 'conviction':...}]
      'support': support(X ∪ Y)
    """
    new_rows = []
    for _, row in rules.iterrows():
        items_base = list(row['antecedents'])
        items_add = list(row['consequents'])
        stat = {
            'items_base': items_base,
            'items_add': items_add,
            'confidence': row['confidence'],
            'lift': row['lift']
        }
        if 'conviction' in row:
            stat['conviction'] = row['conviction']
        merged_items = items_base + items_add
        new_rows.append({
            'items': merged_items,
            'ordered_statistics': [stat],
            'support': row['support']
        })
    df_formatted = pd.DataFrame(new_rows)
    return df_formatted

# 5. Візуалізація: Scatter Plot
def visualize_rules_scatter(rules):
    """
    Будує Scatter Plot:
    - X: support
    - Y: confidence
    - Розмір точки: lift
    - Колір: conviction
    """
    if rules.empty:
        print("\nНемає правил для відображення (rules порожній).")
        return

    x = rules['support'].astype(float)
    y = rules['confidence'].astype(float)
    sizes = rules['lift'].astype(float) * 100
    colors = rules['conviction'].astype(float)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, s=sizes, c=colors, cmap='viridis', alpha=0.6, edgecolors='k')
    plt.xlabel('Підтримка (support)')
    plt.ylabel('Достовірність (confidence)')
    plt.title('Асоціативні правила (FP-Growth): Support vs Confidence')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Переконливість (conviction)')
    plt.show()

# 6. Візуалізація: Горизонтальний Bar Chart (ТОП-N)
def visualize_rules_bar(rules, metric='lift', top_n=10):
    """
    Будує Bar Chart для ТОП-N правил за обраною метрикою (за замовчуванням 'lift').
    """
    if rules.empty:
        print("\nНемає правил для відображення (rules порожній).")
        return

    df_sorted = rules.sort_values(by=metric, ascending=False).head(top_n)
    labels = []
    values = []
    for _, row in df_sorted.iterrows():
        A = ','.join(list(row['antecedents']))
        B = ','.join(list(row['consequents']))
        labels.append(f"{A} -> {B}")
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

# 7. Приклад використання
if __name__ == "__main__":
    # 1) Зчитуємо дані та перетворюємо їх
    df = load_dataset("shopping_data.csv")
    transactions = df_to_transactions(df)
    df_encoded = encode_transactions(transactions)

    # 2) Запуск FP-GROWTH (наприклад, підтримка = 3%)
    frequent_itemsets = run_fpgrowth(df_encoded, min_support=0.03)
    print("=== Часті набори (FP-GROWTH) ===")
    print(frequent_itemsets.head(10))

    # 3) Генерація правил (min_confidence=0.4)
    rules = generate_rules(frequent_itemsets, min_confidence=0.4)
    print("\n=== Усі правила асоціації ===")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'conviction']].head(15))

    # 4) Форматований результат
    df_display = format_rules_for_display(rules)
    print("\n=== Форматований результат у стилі items + ordered_statistics ===")
    print(df_display.head(15))

    # 5) Візуалізація
    # Scatter Plot
    visualize_rules_scatter(rules)

    # Горизонтальний Bar Chart (ТОП-10 за lift)
    visualize_rules_bar(rules, metric='lift', top_n=10)
