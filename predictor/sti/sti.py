import pandas as pd
import fbprophet
import warnings
import ta

class STI:
    """
    A wrapper class for computing various stock technical indicators from ohlcv data.    
    """
    
    def __init__(self, ohlcv):
        """
        Parameter(s):
        -------------
          ohlc : pandas.DataFrame
            A pandas dataframe with the following columns: 'open', 'high', 'low',
            'close', 'volume', 'ds'
        """
        self.ohlcv = ohlcv.copy()
    
    def obv(self):
        """
        Computes the on balance volume. Relates price and volume in the stock market
        based on a cumulative total volume.

        https://www.investopedia.com/terms/o/onbalancevolume.asp
        """
        return ta.volume.OnBalanceVolumeIndicator(self.ohlcv.loc[:, 'close'], 
                                                  self.ohlcv.loc[:, 'volume'],
                                                  fillna=True).on_balance_volume()
  
    def rsi(self, n):
        """
        Computes the relative strength index.

        https://www.investopedia.com/terms/r/rsi.asp
        """
        return ta.momentum.RSIIndicator(self.ohlcv.loc[:, 'close'], 
                                        n=n, 
                                        fillna=True).rsi()

    def sma(self, n):
        """
        Computes the simple moving average.

        https://www.investopedia.com/terms/s/sma.asp#:~:text=Key%20Takeaways-,A%20simple%20moving%20average%20(SMA)%20calculates%20the%20average%20of%20a,a%20bull%20or%20bear%20trend.
        """
        return ta.trend.SMAIndicator(self.ohlcv.loc[:, 'close'], 
                                     n=n,
                                     fillna=True).sma_indicator()

    def ema(self, n):
        """
        Computes the exponential moving average.

        https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        """
        return ta.trend.ema_indicator(self.ohlcv.loc[:, 'close'], 
                                      n=n,
                                      fillna=True)

    def atr(self, n):
        """
        Computes the average true range.

        https://www.investopedia.com/terms/a/atr.asp
        """
        return ta.volatility.AverageTrueRange(self.ohlcv.loc[:, 'high'], 
                                              self.ohlcv.loc[:, 'low'],
                                              self.ohlcv.loc[:, 'close'],
                                              n=n,
                                              fillna=True).average_true_range()

    def mfi(self, n):
        """
        Computes the money flow index.

        https://www.investopedia.com/terms/m/mfi.asp
        """
        return ta.volume.MFIIndicator(self.ohlcv.loc[:, 'high'],
                                      self.ohlcv.loc[:, 'low'],
                                      self.ohlcv.loc[:, 'close'], 
                                      self.ohlcv.loc[:, 'volume'],
                                      n=n,
                                      fillna=True).money_flow_index()

    def adx(self, n):
        """
        Computes the average directional index.

        https://www.investopedia.com/terms/a/adx.asp
        """
        return ta.trend.ADXIndicator(self.ohlcv.loc[:, 'high'], 
                            self.ohlcv.loc[:, 'low'],
                            self.ohlcv.loc[:, 'close'],
                            n=n,
                            fillna=True).adx()

    def cci(self, n):
        """
        Computes the commodity channel index.

        https://www.investopedia.com/terms/c/commoditychannelindex.asp
        """
        return ta.trend.cci(self.ohlcv.loc[:, 'high'], 
                            self.ohlcv.loc[:, 'low'],
                            self.ohlcv.loc[:, 'close'],
                            n=n,
                            fillna=True)

    def willr(self, look_back_period):
        """
        Computes the Williams %R indicator, which determines where today's closing
        price fell within the range on past 14-day's transaction. A type of 
        momentum indicator that moves between 0 and -100. Measures overbought and
        oversold levels. May be used to find entry and exit points in the market.
        'lbp' = lookback period

        https://www.investopedia.com/terms/w/williamsr.asp
        """
        return ta.momentum.WilliamsRIndicator(self.ohlcv.loc[:, 'high'], 
                                              self.ohlcv.loc[:, 'low'],
                                              self.ohlcv.loc[:, 'close'], 
                                              lbp=look_back_period,
                                              fillna=True).wr()

    def trix(self, n):
        """
        Computes the triple exponential moving average.

        https://www.investopedia.com/terms/t/triple-exponential-moving-average.asp
        """
        return ta.trend.TRIXIndicator(self.ohlcv.loc[:, 'close'], 
                                      n=n, 
                                      fillna=True).trix()

    def macd(self, n_slow, n_fast, n_sign):
        """
        Computes the moving average convergence divergence.
        A trend-following momentum indicator that shows the relationship between two moving averages of prices

        n_fast = n period short-term
        n_slow = n period long-term
        n_sign = n period to signal

        https://www.investopedia.com/terms/m/macd.asp
        """
        return ta.trend.MACD(self.ohlcv.loc[:, 'close'],
                             n_slow=n_slow, 
                             n_fast=n_fast, 
                             n_sign=n_sign,
                             fillna=True).macd()

    def bb_low(self, n, ndev=2):
        """
        Computes the lower bollinger band.

        https://www.investopedia.com/terms/b/bollingerbands.asp
        """
        return ta.volatility.BollingerBands(self.ohlcv.loc[:, 'close'],
                                            n=n,
                                            ndev=ndev,
                                            fillna=True).bollinger_lband()
    
    def bb_mid(self, n):
        """
        Computes the middle bollinger band.

        https://www.investopedia.com/terms/b/bollingerbands.asp
        """
        return ta.volatility.BollingerBands(self.ohlcv.loc[:, 'close'], 
                                            n=n,
                                            fillna=True).bollinger_mavg()

    def bb_high(self, n, ndev=2):
        """
        Computes the high bollinger band.

        https://www.investopedia.com/terms/b/bollingerbands.asp
        """
        return ta.volatility.BollingerBands(self.ohlcv.loc[:, 'close'], 
                                            n=n,
                                            ndev=ndev,
                                            fillna=True).bollinger_hband()

    def roc(self, n):
        """
        Computes the rate of change.

        https://www.investopedia.com/terms/a/adx.asp
        """
        return ta.momentum.roc(self.ohlcv.loc[:, 'close'], 
                               n=n, 
                               fillna=True)

    def tsf(self, n, f, return_prophet_only=False):
        """
        Computes a times series forecasting with with prophet.
        """
        prices = self.ohlcv[['ds', 'close']].rename(columns = {'close':'y'})
        prophet = fbprophet.Prophet().fit(prices)
        if return_prophet_only:
            return prophet
        return prophet.predict(prophet.make_future_dataframe(periods=n, freq=f))

    def get_best(self, indicator, high, periods=1):
        """
        Find a value in the range [1, high] that causes `indicator` to be most 
        correlated with the price at the next timestep.
        """
        clse_vals = self.ohlcv.close.shift(periods)
        best_corr = float('-inf')
        best_vals = None
        for i in range(1, high+1):
            try:
                indc_vals = indicator(i)
                curr_corr = abs(pd.DataFrame({
                    'indc' : indc_vals,
                    'clse' : clse_vals
                }).corr().iloc[0, 1])
                if curr_corr > best_corr:
                    best_vals = indc_vals
                    best_corr = curr_corr
            except Exception as e:
                continue
        return best_vals

    def compute_all(self, high=100, macd_nslow=26, macd_nfast=12, macd_signl=9, show_warnings=True):
        """
        Computes all technical indicators and returns a dataframe where the columns
        represent the name of the technical indicator and the values are the actual 
        value of the indicator. To find a suitable value number of periods for the
        indicators, a mini brute-force search over the range [1, high] is performed 
        and the argument most correlated with the price at the next time step is 
        selected.
        """
        if not show_warnings: warnings.filterwarnings('ignore')
        df = pd.DataFrame({
            
            # This is causing a lot of problems, so it'll be left out for now
            # "adx"     : self.get_best(self.adx, high),
            
            "atr"     : self.get_best(self.atr, high),
            "bb_low"  : self.get_best(self.bb_low, high),
            "bb_mid"  : self.get_best(self.bb_mid, high),
            "bb_high" : self.get_best(self.bb_high, high),
            "cci"     : self.get_best(self.cci, high),
            "ema"     : self.get_best(self.ema, high),
            "macd"    : self.macd(macd_nslow, macd_nfast, macd_signl),
            "mfi"     : self.get_best(self.mfi, high),
            "roc"     : self.get_best(self.roc, high),
            "rsi"     : self.get_best(self.rsi, high),
            "trix"    : self.get_best(self.trix, high),
            "willr"   : self.get_best(self.willr, high)
        })
        if not show_warnings: warnings.filterwarnings('always')
        return df.dropna(axis=1)
