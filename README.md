# BTC Predictor
We are the Anchain.ai Bitcoin Price Prediction team from UC Berkeley's Data-X course. Our product uses a machine learning model to help Bitcoin investors make more reliable investment decisions while minimizing risk. 

![BTC Predictor Demo](readme_files/demo.gif)

## Organization and Documentation

### Folder Hierarchy

- **heroku-script**
    - **script.py**: contains the code that schedules when the data should be collected and stored in our Firebase database
    - **util**
        - **database.py**: establishes a connection to our Firebase database
        - **helpers.py**: contains several helper methods for scheduling and timing
        - **logger.py**: contains an abstract class that has two static methods: `log` and `store`. Every logger inherits from this class
        - **news_logger.py**: contains a python class for logging news data from [IEXCloud](https://iexcloud.io/) every 5 minutes. This data is stored in Firebase at the end of each day
        - **reporter.py**: contains a class for sending emails. Each email contains a report that summarizes the data collected for the past day. The reporter also sends an email whenever an error occurs during data collection
        - **tckr_logger.py**: contains a class for logging 5 second Bitcoin exchange data. This data is stored in Firebase at the end of each day
        - **trxn_logger.py**: contains a class for logging Bitcoin transaction data. This data is stored in Firebase at the end of each day

- **predictor**: contains the machine learning models and dashboard for the project
    - **datastore**: contains several files that interact with the data collected from our 3rd party API’s
        - **archives.py**: contains a class with methods that retrieve data from long-term Firebase database
        - **btcstock.py**: contains a class that provides easier access to minute-by-minute OHLCV Bitcoin data from [Bitstamp](https://www.bitstamp.net)
        - **datastore.py**: a class that 
        - **realtime.py**: contains a class with methods that retrieve data from our realtime Firebase database
        - **tweety.py**: contains a class with several methods that allow for fast tweet scraping using [Twint](https://github.com/twintproject/twint)

    - **models**: contains 3 different LSTM models each of which can make minutely Bitcoin price predcitions
        - **baseline.py**: uses the past price as the only feature
        - **indicator.py**: uses technical indicators computed from open-high-low-close-volume (OHLCV) data as features
        - **oracle.py**: uses technical indicators and twitter sentiment scores as features
    - **sti**
        - **sti.py**: a wrapper class for computing various stock technical indicators from OHLCV data
    - **BaselineDemo.ipynb**:
    - **IndicatorDemo.ipynb**:
    - **OracleDemo.ipynb**:
    - **dashboard.ipynb**:
    - **RealTimeTradingSignalDemo.ipynb**:
    

## Technical Sophistication
![Alt text](https://github.com/Bitcoin-Price-Prediction/btcpredictor/blob/main/readme_files/System%20Architecture.png)

## Efficiency

## Reproducibility

### Package Requirements ###
tensorflow==2.3.1
sklearn==0.23.2
plotly==4.9.0
pyrebase==3.0.27
pandas==1.0.1
ta==0.7.0
fbprophet==0.7.1
textblob==0.15.3
twint==2.1.21
requests==2.11.1
numpy==1.18.5
jupyter_dash==0.3.1
dash_bootstrap_components==0.10.7
dash_core_components==1.14.1
dash_html_components==1.1.1
dash==1.18.1

### How to Run: Demos ### 
1. Download required packages
2. Open the demo of your choosing in a Jupyter Notebook (NOT Google Colab)
    1. Note: OracleDemo.ipynb may show warnings due to our twint dependency and takes a long time to warm up. It is recommended to start with IndicatorDemo.ipynb first.
3. Run the cells in order and please follow the comments!


### How to Run: Dashboard ### 
1. Download required packages
2. Open dashboard.ipynb in a Jupyter Notebook (NOT Google Colab)
    1. Note: OracleDemo.ipynb may show warnings due to our twint dependency and takes a long time to warm up.
3. Run the cells in order up until “Example Usage”
4. Run the following cell 
5. Run the cell below with predictor.has_model() until the output is True
6. Run the remaining cells in the notebook
7. Click the link to the external Dash application


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


