import datastore.realtime as realtime
import datastore.archives as archives
import datastore.btcstock as btcstock
import datastore.tweety as tweety
import datetime
import pyrebase

# TODO:
# 	1. Some things can probably be refactored / written more efficiently
#
class DataStore:
	"""
	A wrapper class for accessing realtime stock data and archived stock data.
	"""

	_fb, _rt, _db = None, None, None

	def __init__(self
		, config=None
		, realtime_prices=False
		, btc_cache_size=1440
		, realtime_tweets=False
		, query='bitcoin'):
		"""
		Parameter(s):
		-------------
			config : dict
				A dictionary containing valid Firebase credentials. If specified,
				grants access to realtime and archived database data.

			realtime_prices : bool
				If True, spawns a thread that caches prices. and a thread that caches
				tweets. Upon starting, the price cache will start logging points starting
				at the current time. 

			btc_cache_size : int
				The maximum number of minutely data points to store (default is one day).
				Only has an affect if `realtime_mode` is True.

			realtime_tweets : bool
				If True, spawns a thread that caches tweets. Upon starting, the tweet
				cache will load all tweets since the beginning of today (i.e. 00:00:00)
				to the current time.

			query : string
				The search term for collecting realtime tweets. Only has an effect if 
				`realtime_mode` is True.

		Notes:
		------
			At the time of this writing, there is a bug with the BitStamp API that appears
			at random times throughout the day. If you query for prices in the past eleven
			minutes or so from the current time, the resulting data will not contain the 
			second most recent data point. This data point will appear once enough time has
			passed, but it will not be available at the current time we need it. If you write
			code to detect the missing data point and try to query the API at the same moment 
			in time to recover it, you will receive no data back. This bug does not seem to 
			appear for less recent times. Usually, one data point does not make a difference, 
			but for real time predictions, this data point has a substantial impact on minutely
			predictions since it is very recent. To help reduce data loss, we start a thread to
			cache prices each minute once the first instance of this class is created.
		"""
		self.btcstock = btcstock.BtcStockHandler(realtime_prices, btc_cache_size)
		self.twitter = tweety.TweetScraper(realtime_tweets, query)
		if config is not None:
			DataStore._fb = pyrebase.initialize_app(config)
			DataStore._rt = DataStore._fb.database()
			DataStore._db = DataStore._fb.storage()
			self.archives = archives.ArchivesHandler()
			self.realtime = realtime.RealtimeHandler()

	def _daterange(start, final):
	    yield from [start + datetime.timedelta(days=d) for d in range(0, int((final - start).days))]