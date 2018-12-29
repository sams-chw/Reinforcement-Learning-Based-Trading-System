import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import os, argparse
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


def get_data(tickers, start, end):
    if not os.path.exists('stock_data'):
        os.makedirs('stock_data')
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_data/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_data/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(tickers_list):
    tickers_list = sorted(tickers_list)
    print("\nList of stocks: {}\n".format(tickers_list))

    make_dir('stock_data')
    make_dir('stock_data_plots')

    start_dt = dt.datetime(2010, 12, 2)
    end_dt = dt.datetime(2018, 12, 1)

    # Downloading data from yahoo API
    get_data(tickers_list, start_dt, end_dt)

    for ticker in tickers_list:
        df = pd.read_csv('stock_data/{}.csv'.format(ticker), parse_dates=True, index_col=0)
        df['60ma'] = df['Adj Close'].rolling(window=60, min_periods=0).mean()
        df_col = df['Adj Close'].copy()

        fig = plt.figure()
        plt.figure(figsize=(10, 7))
        plt.plot(df.index, df['Adj Close'], 'b-', label='Adj Close')
        plt.plot(df.index, df['60ma'], 'g-', label='60ma')
        plt.legend(loc='upper left')
        plt.ylabel("stock_price")
        plt.xlabel("Date")
        plt.xticks(rotation=60)
        plt.title("Stock Data by Adj Close: {}".format(ticker))
        plt.savefig("stock_data_plots/stock_data_{}.eps".format(ticker))
        plt.close(fig)

    for count, ticker in enumerate(tickers_list):
        df = pd.read_csv('stock_data/{}.csv'.format(ticker))
        if count == 0:
            split_index = int(0.5 * len(df))
        df_test = df.iloc[split_index:, :].copy()
        df_test['Date'] = pd.to_datetime(df['Date'])
        df_test.set_index('Date', inplace=True)
        df_test['60ma'] = df_test['Adj Close'].rolling(window=60, min_periods=0).mean()

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(df_test.index, df_test['Volume'], align='center', alpha=0.5, color='b', width=0.6)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Volume')
        for tick in ax1.get_xticklabels():
            tick.set_rotation(60)
        ax2 = ax1.twinx()
        ax2.plot(df_test.index, df_test['60ma'], 'g-', label='60ma (Adj Close)')
        ax2.set_ylabel('60ma (Adj Close)')
        fig.tight_layout()
        plt.title("Stock Data by Volume (Test Data): {}".format(ticker))
        plt.legend(loc='upper left')
        plt.savefig("stock_data_plots/stock_data_volume_{}.eps".format(ticker))
        plt.close(fig)
    print()
    print("Plots are generated and saved in the directory.")
    print("Process Completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--symbol', nargs='+', default=['MSFT', 'AMZN', 'IBM'])
    args = parser.parse_args()
    main(args.symbol)


