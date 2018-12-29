import pickle, time, argparse
import sys, itertools
import numpy as np
import datetime as dt
from envs import TradingEnv
from agent import DDQNAgent
from utils import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode', type=int, default=1000,
                      help='number of episode to run')
  parser.add_argument('-b', '--batch_size', type=int, default=32,
                      help='batch size for experience replay')
  parser.add_argument('-i', '--initial_invest', type=int, default=10000,
                      help='initial investment amount')
  parser.add_argument('-s', '--symbol', nargs='+', default=['MSFT', 'AMZN', 'IBM'])
  
  args = parser.parse_args()

  make_dir('weights')
  make_dir('results')

  timestamp = time.strftime('%Y%m%d%H%M')
  start_dt = dt.datetime(2010, 12, 2)
  end_dt = dt.datetime(2018, 12, 1)

  tickers_list = args.symbol
  tickers_list = sorted(tickers_list)
  print("\nYou are investing on the following stocks: {}\n".format(tickers_list))

  # Downloading data from yahoo API
  get_data(tickers_list, start_dt, end_dt)

# Data preprocessing 
  col = 'Adj Close'
  full_data, data = compile_data(tickers_list, col)
  data = np.around(data)
  data_split = int(0.5 * len(data[0]))

  date_split = full_data[0].index.values[data_split]
  date_split = str(date_split)[:10]
  print("\nTraing and Testng date split: ", date_split[:10])
   
  train_data = data[:, :data_split]
  print("\ntrain data ->")
  print(train_data)
  print("(dimensions, elements):", train_data.shape)
  test_data = data[:, data_split:]
  print("\ntest data ->")
  print(test_data)
  print("(dimensions, elements):", test_data.shape)

  full_train_data = []
  full_test_data = []

  for i, k in enumerate(tickers_list):
    print("\nStock data for {} (first 5 rows):".format(tickers_list[i]))
    print(full_data[i].head(5))
    print("(rows, columns):", full_data[i].shape)
    full_train_data.append(full_data[i].iloc[:data_split,:])

# Training phase

  print("\n===========================  Training Mode  ======================================")

  env = TradingEnv(train_data, args.initial_invest)
  state_size = env.observation_space.shape
  print("state size", state_size)
  action_size = env.action_space.n
  print("action size", action_size)
  trade_agent = DDQNAgent(state_size, action_size)
  scaler = get_scaler(env)

  labels = ['episode', 'total_reward']
  results = {x: [] for x in labels}
  acts = []
  portfolio_value = []

  train_results = {}

  for e in range(args.episode):
    state = env.reset()
    score = 0
    total_profit = 0
    state = scaler.transform([state])

    for time in range(env.n_step):
      action = trade_agent.act(state)
      next_state, reward, done, info = env.step(action)

      if e == args.episode-1:
        acts.append(env.action_vector_new)

      score += reward
      total_profit += info['profit']
      next_state = scaler.transform([next_state])
      
      trade_agent.remember(state, action, reward, next_state, done)  # when in training phase

      state = next_state

      if done:
        score = np.round(score/1000, 2)
        print("Episode: {},  Current portfolio value: {}, total_profit: {}, Score: {}.".format(
              e, info['cur_val'], total_profit, score))
        portfolio_value.append(info['cur_val']) # append episode end portfolio value
        if e == args.episode - 1:
          train_results['score'] = score
          train_results['total_profit'] = total_profit
          train_results['portfolio_value'] = info['cur_val']
        break

      if len(trade_agent.memory) > args.batch_size:   # when in training phase 
        trade_agent.replay(args.batch_size)

    results['episode'].append(e)
    results['total_reward'].append(score)

    if (e+1) % 1 == 0:   # when in training phase 
      trade_agent.model.save("weights/{}-ddqn.h5".format(timestamp))

    sys.stdout.flush()

  Qplot(results)

  for count,ticker in enumerate(tickers_list):
      train_acts = []
      for act in acts:
        train_acts.append(act[count])
      train_acts.append(np.nan)
      full_train_data[count]['Action'] = train_acts

  print("\nTraining completed.")

  pv_train_plot(portfolio_value)

  print("\n=============================  Testing Mode  ========================================")

  # remake the env with test data
  for i, k in enumerate(tickers_list):
    full_test_data.append(full_data[i].iloc[data_split:,:])

  env = TradingEnv(test_data, args.initial_invest)
  state_size = env.observation_space.shape
  action_size = env.action_space.n
  model_name = "weights/{}-ddqn.h5".format(timestamp)
  trade_agent = DDQNAgent(state_size, action_size, True, model_name)
  scaler = get_scaler(env)

  labels = ['episode', 'total_reward']
  results = {x: [] for x in labels}
  acts = []
  portfolio_value = []
  test_results = {}

  for e in range(args.episode):
    state = env.reset()
    score = 0
    total_profit = 0
    state = scaler.transform([state])

    for time in range(env.n_step):
      action = trade_agent.act(state)
      next_state, reward, done, info = env.step(action)
      if e == args.episode-1:
        acts.append(env.action_vector_new)
      score += reward
      total_profit += info['profit']
      next_state = scaler.transform([next_state])
      state = next_state

      if done:
        score = np.round(score/1000, 2)
        print("Episode: {},  Current portfolio value: {}, total_profit: {}, Score: {}.".format(
              e, info['cur_val'], total_profit, score))
        portfolio_value.append(info['cur_val']) # append episode end portfolio value
        if e == args.episode - 1:
          test_results['score'] = score
          test_results['total_profit'] = total_profit
          test_results['portfolio_value'] = info['cur_val']
        break

    results['episode'].append(e)
    results['total_reward'].append(score)
    sys.stdout.flush()

  Qplot(results, True)

  for count, ticker in enumerate(tickers_list):
    test_acts = []
    for act in acts:
      test_acts.append(act[count])
    test_acts.append(np.nan)
    full_test_data[count]['Action'] = test_acts

  print("\nTesting completed.")

  portfolio_value.append(np.nan)
  
#   plot results
  pv_test_plot(portfolio_value, args.initial_invest)
  plot_train_test(full_train_data, full_test_data, tickers_list, date_split)
  plot_train_test_ddqn(full_test_data, tickers_list)
