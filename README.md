# Automated Trading using Reinforcement Learning (DDQN)
## Capstone-Project-MLND-Udacity

This repo contains code for the MLND capstone project 

# Dependencies
Python 2.7. To install all the libraries, run pip install -r requirements.txt

# Table of content
**agent.py:** a Double Deep Q learning agent  
**envs.py:** a multi-stock trading environment  
**utils.py:** some utility functions  
**run.py:** main file for the project  
**run_stock_data.py:** additional script only to generate plots for the stock data  
**requirement.txt:** all dependencies  

# How to run

1. For automated trading, run the following command:  
        >   `python run.py`    
There are some default command-line arguments (see the run.py script for details). For example, the above command is basically same as,  
        >   `python run.py -e 1000 -i 10000 -s MSFT AMZN IBM`  
You can input any number of episodes, amount of initial investment and stocks you want by following the above sequence in the         command-line. For example,  
        >   `python run.py -e 2000 -s NVDA INTC`
        
2. To generate plots for the stock data, run the following command.      
        >   `python run_stock_data.py`  
which is by default same as,  
        >   `python run_stock_data.py -s MSFT AMZN IBM`   
But you can input any stocks you want instead of MSFT, AMZN or IBM. For example,      
        >   `python run_stock_data.py -s TSLA AAPL`
