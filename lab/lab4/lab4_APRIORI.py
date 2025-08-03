import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from adjustText import adjust_text


# Завантаження датасету (зчитування CSV-файлу у вигляді DataFrame)
def load_dataset(filename="shopping_data.csv"):
    return pd.read_csv(filename)


# Перетворення датасету у формат транзакцій
# Кожен рядок представляє список товарів, які були придбані разом
def preprocess_data(dataset):
    transactions = dataset.apply(lambda x: [col for col in dataset.columns if x[col] == 1], axis=1).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)

    return pd.DataFrame(te_ary, columns=te.columns_)


# Виконання алгоритму Apriori
# Пошук частих наборів товарів та генерація правил асоціації
def run_apriori(df, min_support=0.05, min_confidence=0.4):
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    print("\nЧасті набори товарів:")
    print(frequent_itemsets)  # Відладковий вивід

    if frequent_itemsets.empty:
        print("\nНе знайдено частих наборів товарів. Спробуйте знизити min_support.")
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        print("\nНе знайдено сильних правил асоціації. Спробуйте знизити min_confidence.")
        return pd.DataFrame()

    # Обчислення коефіцієнта переконливості (conviction)
    rules["conviction"] = (1 - rules["consequent support"]) / (1 - rules["confidence"])

    return rules


# Візуалізація правил асоціації
def visualize_rules(rules):
    if rules.empty:
        print("\nНемає правил для візуалізації.")
        return

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(rules['support'], rules['confidence'],
                          s=rules['lift'] * 100, alpha=0.5,
                          c=rules['conviction'], cmap='viridis', edgecolors='k')

    plt.colorbar(scatter, label='Переконливість (Conviction)')
    plt.xlabel('Підтримка (Support)')
    plt.ylabel('Достовірність (Confidence)')
    plt.title('Візуалізація правил асоціації')

    texts = []
    for i, row in rules.iterrows():
        label = f"{', '.join(row['antecedents'])} → {', '.join(row['consequents'])}"
        texts.append(plt.text(row['support'], row['confidence'], label, fontsize=8, ha='center', va='bottom'))

    adjust_text(
        texts,
        only_move={'points': 'y', 'texts': 'y'},
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, shrinkA=8),
        autoalign='y',
        force_text=0.1,
        force_points=0.1,
        expand_points=(1.1, 1.1),
        expand_text=(1.1, 1.1),
        avoid_self=True,
        save_steps=False,
        verbose=0
    )

    plt.show()


if __name__ == "__main__":
    dataset = load_dataset()
    df_encoded = preprocess_data(dataset)

    apriori_rules = run_apriori(df_encoded)

    if not apriori_rules.empty:
        print("\nПравила APRIORI:")
        print(apriori_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'conviction']])
        visualize_rules(apriori_rules)
