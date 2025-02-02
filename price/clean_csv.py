import pandas as pd

def clean_csv(input_file, output_file):
    # Загружаем данные из CSV файла
    try:
        data = pd.read_csv(input_file, parse_dates=['Date'], index_col='Date')
    except FileNotFoundError:
        print(f"Файл {input_file} не найден.")
        return
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return

    # Выводим информацию о загруженных данных
    print("Исходные данные:")
    print(data.info())

    # Удаляем столбцы, которые не нужны для обучения
    columns_to_drop = [
        "('Open', 'INDEX')", "('Open', 'MOEX')", "('Close', 'INDEX')", "('Close', 'MOEX')",
        "('Ticker', '')", "('High', 'INDEX')", "('High', 'MOEX')", "('Low', 'INDEX')",
        "('Low', 'MOEX')", "('Adj Close', 'INDEX')", "('Adj Close', 'MOEX')", "('Volume', 'INDEX')",
        "('Volume', 'MOEX')", "('Open', 'RTS')", "('Close', 'RTS')", "('High', 'RTS')",
        "('Low', 'RTS')", "('Adj Close', 'RTS')", "('Volume', 'RTS')", "('Open', 'MICEX10')",
        "('Close', 'MICEX10')", "('High', 'MICEX10')", "('Low', 'MICEX10')", "('Adj Close', 'MICEX10')",
        "('Volume', 'MICEX10')"
    ]
    
    data.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Удаляем строки с NaN значениями
    data.dropna(inplace=True)

    # Выводим информацию о очищенных данных
    print("Очищенные данные:")
    print(data.info())

    # Сохраняем очищенные данные в новый CSV файл
    data.to_csv(output_file)
    print(f"Очищенные данные сохранены в {output_file}")

if __name__ == "__main__":
    input_csv_file = "combined_stock_data.csv"  # Укажите путь к вашему входному файлу
    output_csv_file = "cleaned_stock_data.csv"   # Укажите путь к выходному файлу
    clean_csv(input_csv_file, output_csv_file)
