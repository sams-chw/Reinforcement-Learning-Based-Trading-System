import os
import pandas as pd
import pandas_datareader.data as web
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style
import plotly.graph_objs as go
from plotly.offline import plot

style.use('ggplot')


def get_data(tickers, start, end):
    if not os.path.exists('stock_data'):
        os.makedirs('stock_data')
    for ticker in tickers:
        if not os.path.exists('stock_data/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_data/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

def compile_data(tickers, col='Adj Close'):
    data_all = []
    data_col_all = []

    for ticker in tickers:
        df = pd.read_csv('stock_data/{}.csv'.format(ticker))
        df_col = df[col].copy()
        data_col_all.append(df_col.values)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        data_all.append(df)

    return data_all, np.array(data_col_all)

def get_scaler(env):
    """ Takes a env and returns a scaler for its observation space """
    low = [0] * (env.n_stock * 2 + 1)
    high = []
    max_price = env.stock_price_history.max(axis=1)
    min_price = env.stock_price_history.min(axis=1)
    max_cash = env.init_invest * 3  # 3 is a magic number...
    max_stock_owned = max_cash // min_price
    for i in max_stock_owned:
        high.append(i)
    for i in max_price:
        high.append(i)
    high.append(max_cash)

    scaler = StandardScaler()
    scaler.fit([low, high])
    return scaler

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_train_test(train_data, test_data, tickers, date_split):
    for i in range(len(tickers)):
        data = [
            go.Candlestick(x=train_data[i].index,
                           open=train_data[i]['Open'],
                           high=train_data[i]['High'],
                           low=train_data[i]['Low'],
                           close=train_data[i]['Close'],
                           name='train'),
            go.Candlestick(x=test_data[i].index,
                           open=test_data[i]['Open'],
                           high=test_data[i]['High'],
                           low=test_data[i]['Low'],
                           close=test_data[i]['Close'],
                           name='test')
        ]
        title = 'Stock data for {}'.format(tickers[i])
        layout = {
            'title': title,
            'showlegend': False,
            'shapes': [
                {'x0': date_split, 'x1': date_split, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',
                 'line': {'color': 'rgb(0,0,0)', 'width': 3}}
            ],
            'annotations': [
                {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'left',
                 'text': '         test data'},
                {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'right',
                 'text': 'train data         '}
            ]
        }
        figure = go.Figure(data=data, layout=layout)
        plot(figure, filename='results/Stock_data_{}.html'.format(tickers[i]), auto_open=False)


def plot_train_test_ddqn(test_data, tickers):
    for i in range(len(tickers)):
        test_sell = test_data[i][test_data[i]['Action'] == 0]
        test_hold = test_data[i][test_data[i]['Action'] == 1]
        test_buy = test_data[i][test_data[i]['Action'] == 2]

        Action_color0, Action_color1, Action_color2 = 'gray', 'cyan', 'magenta'

        data = [
            go.Candlestick(x=test_sell.index,
                           open=test_sell['Open'],
                           high=test_sell['High'],
                           low=test_sell['Low'],
                           close=test_sell['Close'],
                           increasing=dict(line=dict(color=Action_color0)),
                           decreasing=dict(line=dict(color=Action_color0)),
                           name='test_sell'),
            go.Candlestick(x=test_hold.index,
                           open=test_hold['Open'],
                           high=test_hold['High'],
                           low=test_hold['Low'],
                           close=test_hold['Close'],
                           increasing=dict(line=dict(color=Action_color1)),
                           decreasing=dict(line=dict(color=Action_color1)),
                           name='test_hold'),
            go.Candlestick(x=test_buy.index,
                           open=test_buy['Open'],
                           high=test_buy['High'],
                           low=test_buy['Low'],
                           close=test_buy['Close'],
                           increasing=dict(line=dict(color=Action_color2)),
                           decreasing=dict(line=dict(color=Action_color2)),
                           name='test_buy')
        ]
        title = 'DDQN model on {} stock test data'.format(tickers[i])
        layout = {
            'title': title,
            'showlegend': True,
        }
        figure = go.Figure(data=data, layout=layout)
        plot(figure, filename='results/Stock_data_{}_DDQN.html'.format(tickers[i]), auto_open=False)

def Qplot(Qresult, test = False):
    plt.figure(figsize=(10, 6))
    plt.plot(Qresult['episode'], Qresult['total_reward'])
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")
    _ = plt.ylim()
    if test:
        plt.title("DDQN Rewards (Test Data)")
        plt.savefig("results/test_rewards.png")
    else:
        plt.title("DDQN Rewards (Train Data)")
        plt.savefig("results/train_rewards.png")

def pv_train_plot(pv_axis):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(pv_axis)), pv_axis, color="blue")
    plt.ylabel("portfolio_value")
    plt.xlabel("episodes")
    _ = plt.ylim()
    plt.title("Portfolio Value for Train Run")
    plt.savefig("results/portfolio_train.png")

def pv_test_plot(pv, init_invest):
    pv_mean = np.mean(pv[:-1])
    pv_mean_list = [pv_mean]*len(pv)
    pv_init = [init_invest]*len(pv)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(len(pv)), pv, color="blue")
    plt.plot(range(len(pv)), pv_mean_list, 'g-', label='average_portfolio_value ({})'.format(
                    int(np.around(pv_mean))))
    plt.plot(range(len(pv)), pv_init, 'r--', label='initial_invest ({})'.format(init_invest))
    plt.legend(loc='upper left')
    plt.ylabel("portfolio_value")
    plt.xlabel("episodes")
    _ = plt.ylim()
    plt.title("Portfolio Value for Test Run")
    plt.savefig("results/portfolio_test.png")
    plt.close(fig)
