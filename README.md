# BTC Predictor
We are the Anchain.ai Bitcoin Price Prediction team from UC Berkeley's Data-X course. Our product uses a machine learning model to help Bitcoin investors make more reliable investment decisions while minimizing risk. 

![BTC Predictor Demo](readme_files/demo.gif)

## Organization and Documentation

Folder Hierarchy

- heroku-script
    - script.py
    - util
        - database.py
        - helpers.py
        - logger.py
        - news_logger.py
        - reporter.py
        - tckr_logger.py
        - trxn_logger.py

- predictor: contains the machine learning models and dashboard for the project
    - datastore: contains several files that interact with the data collected from our 3rd party API’s
        - archives.py
        - btcstock.py
        - datastore.py
        - realtime.py
        - tweety.py
    - models: contains 3 different LSTM models each of which can make minutely Bitcoin price predcitions
        - baseline.py: uses the past price as the only feature
        - indicator.py: uses technical indicators computed from open-high-low-close-volume (OHLCV) data as features
        - oracle.py: uses technical indicators and twitter sentiment scores as features
    - sti
        - sti.py: a wrapper class for computing various stock technical indicators from OHLCV data
    - BaselineDemo.ipynb: 
    - IndicatorDemo.ipynb
    - OracleDemo.ipynb
    - dashboard.ipynb
    

## Technical Sophistication
![Alt text](https://github.com/Bitcoin-Price-Prediction/btcpredictor/blob/main/readme_files/System%20Architecture.png)

## Efficiency

## Reproducibility

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


