# Capstone-Project-MLND
Automated Trading using Reinforcement Learning (DDQN)

This repo contains code for the MLND capstone project 

# Dependencies
Python 2.7. To install all the libraries, run pip install -r requirements.txt

# Table of content
agent.py: a Double Deep Q learning agent
envs.py: a multi-stock trading environment
utils.py: some utility functions
run_stock_data.py: to generate graphs for stock data
run.py: main()
requirement.txt: all dependencies

data/: 3 csv files with IBM, MSFT, and QCOM stock prices from Jan 3rd, 2000 to Dec 27, 2017 (5629 days). The data was retrieved using Alpha Vantage API

# How to run
To train a Deep Q agent, run python run.py --mode train. There are other parameters and I encourage you look at the run.py script. After training, a trained model as well as the portfolio value history at episode end would be saved to disk.

To test the model performance, run python run.py --mode test --weights <trained_model>, where <trained_model> points to the local model weights file. Test data portfolio value history at episode end would be saved to disk.
