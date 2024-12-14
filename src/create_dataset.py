import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def create_dataset(tickers):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 15)

    all_data = []

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date.strftime("%Y-%m-%d"))

        data["Ticker"] = ticker

        all_data.append(
            data[["Open", "Close", "Ticker", "High", "Low", "Adj Close", "Volume"]]
        )

    combined_data = pd.concat(all_data)

    combined_data.reset_index(inplace=True)

    return combined_data


tickers = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "AMZN",  # Amazon.com, Inc.
    "GOOGL",  # Alphabet Inc. (Class A)
    "FB",  # Meta Platforms, Inc. (Facebook)
    "TSLA",  # Tesla, Inc.
    "NVDA",  # NVIDIA Corporation
    "PYPL",  # PayPal Holdings, Inc.
    "NFLX",  # Netflix, Inc.
    "INTC",  # Intel Corporation
    "CSCO",  # Cisco Systems, Inc.
    "CMCSA",  # Comcast Corporation
    "PEP",  # PepsiCo, Inc.
    "AVGO",  # Broadcom Inc.
    "TXN",  # Texas Instruments Incorporated
    "QCOM",  # Qualcomm Incorporated
    "AMGN",  # Amgen Inc.
    "SBUX",  # Starbucks Corporation
    "ADBE",  # Adobe Inc.
    "INTU",  # Intuit Inc.
    "MDLZ",  # Mondelez International, Inc.
    "ISRG",  # Intuitive Surgical, Inc.
    "CHKP",  # Check Point Software Technologies Ltd.
    "GILD",  # Gilead Sciences, Inc.
    "ATVI",  # Activision Blizzard, Inc.
    "BKNG",  # Booking Holdings Inc.
    "LRCX",  # Lam Research Corporation
    "NOW",  # ServiceNow, Inc.
    "FISV",  # Fiserv, Inc.
    "SPLK",  # Splunk Inc.
    "ZM",  # Zoom Video Communications, Inc.
    "DOCU",  # DocuSign, Inc.
    "SNPS",  # Synopsys, Inc.
    "MRNA",  # Moderna, Inc.
    "BIIB",  # Biogen Inc.
    "ILMN",  # Illumina, Inc.
    "MELI",  # MercadoLibre, Inc.
    "COST",  # Costco Wholesale Corporation
    "JD",  # JD.com, Inc.
    "PDD",  # Pinduoduo Inc.
    "BIDU",  # Baidu, Inc.
    "NTES",  # NetEase, Inc.
    "NTGR",  # NETGEAR, Inc.
    "ASML",  # ASML Holding N.V.
    "MOEX Index",  # Индекс Московской биржи
    "RTS Index",  # Индекс РТС
    "MICEX10 Index",  # Индекс ММВБ-10
]

dataset = create_dataset(tickers)

home_dir = os.path.expanduser("~")
# file_path = os.path.join(home_dir, "IPSA", "combined_stock_data.csv")
file_path = "combined_stock_data.csv"

dataset.to_csv(file_path, index=False)

print("Dataset created and saved as 'combined_stock_data.csv'.")
