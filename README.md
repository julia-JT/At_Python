# At_Python
Python_Att
Task:
"""Возьмите датасет «Финансовая отчётность российских компаний» и
проанализируйте его с помощью Pandas или Polars.
ВАЖНО: Перед началом анализа сократите количество колонок до 20 и
количество строк до 1000. Также, можете использовать streaming, чтобы
не скачивать сразу весь датасет."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from itertools import islice  # Для ограничения строк

# Загружаем датасет с streaming
# Шаг 1: Загрузка датасета с streaming и ограничение до 1000 строк
print("Загрузка датасета...")
dataset = load_dataset("irlspbru/RFSD", split="train", streaming=True)
df = pd.DataFrame(list(dataset.take(1000)))  # Берем первые 1000 строк
print(f"Загружено {len(df)} строк.")

# Шаг 2: Переименование столбцов с помощью справочника
print("Переименование столбцов...")
renaming_df_url = 'https://raw.githubusercontent.com/irlcode/RFSD/main/aux/descriptive_names_dict.csv'
renaming_df = pd.read_csv(renaming_df_url)
renaming_dict = {orig: new for orig, new in zip(renaming_df['original'], renaming_df['descriptive']) if orig in df.columns}
df.rename(columns=renaming_dict, inplace=True)
#print("Переименованные столбцы:", list(renaming_dict.values()))

# Шаг 3: Выбор 17 значимых колонок из списка (адаптируйте, если названий нет)
significant_columns = ['year','ogrn','region','age','B_fixed_assets','B_current_assets','B_accounts_receivable','B_inventories','B_total_equity',
                       'B_charter_capital','B_reserve_capital',
'B_assets','PL_revenue','PL_gross_profit','PL_income_tax','PL_tax_liab','PL_net_profit','PL_reval','CFi_loans','PU_income_activities'] 
# Ограничено до 17; если названий нет, замените на реальные из df.columns
df = df[[col for col in significant_columns if col in df.columns]]
print(f"Выбрано {len(df.columns)} колонок из 17.")

# Шаг 4: Добавление новых колонок
print("Добавление новых колонок...")
# ROA: net_profit / assets
df['ROA'] = df['PL_net_profit'] / df['B_assets']
df['ROE'] = df['PL_net_profit'] / df['B_total_equity']

print(df.info())
# Анализ
# 1. Описательная статистика
print("Описательная статистика:")
print(df.describe())
# 2. Построение гистограмм и анализ выбросов
# Для примера выберем первые 3 числовые колонки
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
hist_cols = numeric_cols[:3]

for col in hist_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col].dropna(), bins=5, kde=True)
    plt.title(f"Гистограмма и KDE для {col}")
    plt.show()

    # Анализ выбросов через межквартильный размах (IQR)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Колонка {col}: выбросов найдено {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
#3. Матрица корреляций и визуализация
plt.figure(figsize=(12,10))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Матрица корреляций")
plt.show()

# Найдем пары с высокой корреляцией (например, |r| > 0.7)
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i,j]
        if abs(r) > 0.7:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], r))

print("Пары колонок с высокой корреляцией (>0.7 по модулю):")
for c1, c2, r in high_corr:
    print(f"{c1} и {c2}: r = {r:.2f}")

 Пары колонок с высокой корреляцией (>0.7 по модулю):
B_fixed_assets и B_current_assets: r = 0.85
B_fixed_assets и B_accounts_receivable: r = 0.83
B_fixed_assets и B_assets: r = 0.91
B_fixed_assets и PL_revenue: r = 0.87
B_fixed_assets и PL_gross_profit: r = 0.76
B_current_assets и B_accounts_receivable: r = 0.98
B_current_assets и B_inventories: r = 0.81
B_current_assets и B_assets: r = 0.98
B_current_assets и PL_revenue: r = 0.93
B_current_assets и PL_tax_liab: r = 0.95
B_accounts_receivable и B_assets: r = 0.96
B_accounts_receivable и PL_revenue: r = 0.95
B_accounts_receivable и PL_tax_liab: r = 0.96
B_inventories и B_reserve_capital: r = 1.00
B_inventories и B_assets: r = 0.80
B_inventories и PL_tax_liab: r = 0.99
B_total_equity и PL_tax_liab: r = 1.00
B_assets и PL_revenue: r = 0.93
B_assets и PL_tax_liab: r = 0.84
PL_revenue и PL_gross_profit: r = 0.80
PL_revenue и PL_income_tax: r = -0.84
PL_revenue и PL_tax_liab: r = 1.00
PL_gross_profit и PL_income_tax: r = -0.98
PL_gross_profit и PL_tax_liab: r = 1.00
PL_income_tax и PL_tax_liab: r = -1.00
PL_income_tax и PL_net_profit: r = -0.75
PL_tax_liab и PL_net_profit: r = 0.99
PL_tax_liab и ROA: r = 0.85

#4. Диаграммы рассеивания для коррелирующих колонок
for c1, c2, r in high_corr:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=c1, y=c2, alpha=0.6)
    print(plt.title(f"Scatter plot: {c1} vs {c2} (r={r:.2f})"))
    print(plt.show())

''' 5.Выводы по анализу:
1) Датасет содержит много пустых строк по разным столбцам, 
для машинного обучения необходимо будет или дополнять эти столбцы данными (усредненными или иначе). Есть дубли столбцов по смыслу
2) Из гистограммы возраста видно, то основная масса компаний возрастом 3-5 лет, чем больше возраст, тем число компаний падает
3) Из гистограммы основных средств видно, что у большей части компаний они не превышают 100 000
4) Из гистограммы оборотных активов видно, что их объем также коррелируется
5) Под тепловой картой выведены наиболее коррелируемые колонки.
6) На диаграммах рассеивания видны точечные выбросы в разрезе коррелируемых колонок
Аномалия - Выбросы в total_assets (например, компании с активами > 1e9 могут быть гигантами или ошибками в данных)
Закономерность - Высокая корреляция между revenue и net_profit (ожидаемо, так как прибыль зависит от доходов)
Неожиданная связь - Низкая корреляция между ROA и ROE может говорить о наличии заемных средств, но столбец с кредитами чаще содержит пустые значения'''
