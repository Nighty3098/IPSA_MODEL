import pandas as pd

# Задайте путь к вашему CSV файлу
input_file = "data.csv"  # замените на имя вашего файла
output_file = "output.csv"  # имя файла для сохранения очищенных данных

# Чтение CSV файла
df = pd.read_csv(input_file)

# Оставляем только нужные колонки
columns_to_keep = ["Ticker", "Date", "Close", "High", "Low", "Open", "Volume"]
df_cleaned = df[columns_to_keep]

# Удаляем строки с пустыми полями
df_cleaned = df_cleaned.dropna()

# Сохранение очищенного DataFrame в новый CSV файл
df_cleaned.to_csv(output_file, index=False)

print(f"Очищенный файл сохранен как {output_file}")
