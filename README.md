# Bitcoin Predictor
We are the Anchain.ai Bitcoin Price Prediction team from UC Berkeley's Data-X course. 

![BTC Predictor Demo](readme_files/demo.gif)

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
![Alt text](readme_files/System Architecture.png?raw=true "Optional Title")



