# Bitcoin Predictor
We are the Anchain.ai Bitcoin Price Prediction team from UC Berkeley's Data-X course 

![BTC Predictor Demo](readme_files/demo.gif)

## About the Team
![](https://github.com/Bitcoin-Price-Prediction/btcpredictor/blob/main/readme_files/Team.png)

- We are a team of UC Berkeley students from various backgrounds and majors 
- **Chris De Leon**
    - Major: Computer Science
    - About: Has past research and internship experience in Machine Learning and AI
    - Project Responsibilities: Creating and implementing various aspects of the backend model 
- **Michela Burns** 
    - Major: Data Science
    - About: Has past project and internship experience in Machine Learning and Data Analytics. 
    - Project Responsibilities: Twitter web scraping and real-time dashboard
- **Vivian Lin** 
    - Major: Data Science and Economics 
    - About: Has a background in behavioral economics, finance, and three years of stock trading experience on the side
    - Project Responsibilities: Creating and implementing the trading strategies

## Purpose
Our product uses a machine learning model to help Bitcoin investors make more reliable investment decisions while minimizing risk. 

## Problem
When investing in Bitcoin, there are a lot of things that can get in an investor’s way
- **Price Volatility**: One major hurdle that investors must account for is Bitcoin’s extreme price volatility
- **Human Irrationality**: The cryptocurrency market is largely susceptible to market sentiment and collective behaviors
- **Financial Risk**: Investing in Bitcoin can be high-risk, and with high-risk investing comes the possibility of losing money

## Solution 
BTC Predictor is a dashboard that displays real-time...
- **Bitcoin Price Forecasts**
- **Trading Signals**
- **Bitcoin Twitter Sentiment**

## Approach
Our approach consisted of 4 major parts.
1. **Research**
    1. Our mentor provided us with high quality research papers that helped point us to several high-performing models to explore
2. **Big Data Collection**
    1. We scraped over two years worth of tweets and built our own sensor for logging high resolution stock data
    2. This data was crucial for backtesting and tuning our models
3. **Modeling**
    1. Our model uses a mixture of stock technical indicators and Twitter data as features to account for price fluctuations and human irrationality
4. **Trading Signals**
    1. We used 15 trading strategies, which consists of 10 EMA crossover strategies, 2 Ichimoku Cloud-based strategies, and 3 momentum-based strategies to create our overall trading signal gauge
    2. Each of these 15 strategies would output a buy or sell signal in real-time
    3. Then, we took the average trading signals and converted it into an overall gauge of trading signal that ranges from strong sell to strong buy
  
## Features
### OHLC
- Open, high, low, close
- On balance volume
- Relative strength index
- Simple moving average
- Exponential moving average
- Average true range
- Money flow index
- Commodity channel index
- Williams %R
- Triple exponential moving average
- Moving average convergence divergence
- Bollinger bands (low, middle, high)
- Rate of change
### Twitter
- Average subjectivity score
- Average polarity score
- Total subjectivity score
- Total polarity score
- Number of tweets per minute
- Most common tone

## System Architecture
![Alt text](https://github.com/Bitcoin-Price-Prediction/btcpredictor/blob/main/readme_files/System%20Architecture.png)

## Model
- **What type of model are you using?**
    - Our product is powered by an LSTM model
- **How did you develop your model?**
    - We performed exploratory data analysis and time series correlation analysis on our Twitter and OHLC data with varying time ranges, such as a day, week, and month
- **How well does the model perform?**
    - Direction Accuracy: We’ve observed that the model can correctly predict the direction of a price change AT LEAST 85% of the time if used every 10 to 15 minutes
    - SMAPE: We used the Symmetric Mean Absolute Percentage Error (SMAPE) to get a percentage bounded between 0 and 100 that quantifies how far off our predictions are with the actual price. From our experiments, we found that this was consistently below 1%, which indicates that the predicted price is of a similar magnitude to the actual price
