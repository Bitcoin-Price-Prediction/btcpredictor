import datastore.datastore as datastore
import pandas as pd

class RealtimeHandler:
	"""
	This class allows you to access snapshots of the realtime database.
	Stock data is collected every 5 seconds, and news data is collected 
	every 5 minutes. A dataframe containing all data in the database is
	returned on every call. You may wish to store the data after each 
	call so that the Firebase monthly download limit isn't exceeded.
	"""

	def news(self):
		"""
		Returns a dataframe containing all stock data currently in the realtime
		database. The members (as given on IEXCloud) are:

			datetime 	: Millisecond epoch of time of article.
			hasPaywall	: Whether the news source has a paywall.
			headline	: The title of the article.
			image 		: URL to IEX Cloud for associated news image. 
			lang 		: Language of the source article.
			related 	: Comma-delimited list of tickers associated with this news article. 
			source 		: Source of the news article. Make sure to always attribute the source.
			summary		: A brief description of the article.
			url 		: URL to IEX Cloud for associated news image. 

		The following member is also included:

			LAtime		: The time the article was logged in the database.
		"""
		news = datastore.DataStore._rt.child('News').get().val().values()
		data = pd.DataFrame()
		for item in news:
		    temp = pd.DataFrame(item['articles'])
		    temp['LAtime'] = item['LAtime']
		    data = pd.concat([data, temp], ignore_index=True)
		return data

	def stocks(self):
		"""
		Returns a dataframe containing all stock data currently in the realtime
		database. The members (as given on Bitstamp) are:

			ask		: Lowest sell order.
			bid		: Highest buy order.
			high 		: Last 24 hours price high.
			last 		: Last BTC price.
			low 		: Last 24 hours price low.
			open 		: First price of the day.				
			timestamp 	: Unix timestamp date and time.
			volume		: Last 24 hours volume.
			vwap 		: Last 24 hours volume weighted average price.

		The following member is also included:
		
			LAtime		: The time the article was logged in the database.

		Notes:
		------
			Due to timing errors, data from the previous day may leak into the next day.
			Keep this in mind when working with the data.
		"""
		return pd.DataFrame(datastore.DataStore._rt.child('Stocks').get().val().values())