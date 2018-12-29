# Capstone-Project-MLND
Automated Trading using Reinforcement Learning (DDQN)

This repo contains code for the MLND capstone project 

# Dependencies
Python 2.7. To install all the libraries, run pip install -r requirements.txt

# Table of content
agent.py: a Double Deep Q learning agent

envs.py: a multi-stock trading environment

utils.py: some utility functions

run_stock_data.py: additional script only to generate plots for the stock data

run.py: main file for the project

requirement.txt: all dependencies

data/: 3 csv files with IBM, MSFT and AMZN stock data from 12/1/2010 to 11/30/2018. The data can be download using Yahoo Finance API

# How to run

For automated trading, run the following command:

python run.py 

There are some default command-line arguments. Please see the run.py script. For example, the above command is basically same as below.

python run.py -e 1000 -i 10000 -s MSFT AMZN IBM

You can input the number of episodes, amount of initial investment and the stocks as you desire by following the above sequence in the command-line.

For example, 
python run.py -2000 -s NVDA INTC

For generating plots for the stock data, run the following command:

python run_stock_data.py

which is by default same as

python run_stock_data.py -s MSFT AMZN IBM 

But you can input any stocks you want instead of MSFT, AMZN or IBM. For example,
python run_stock_data.py -s TSLA AAPL
