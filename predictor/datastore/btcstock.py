import datastore.datastore as datastore
import pandas as pd
import collections
import threading
import calendar
import requests
import json
import time
import os

from datetime import datetime, timedelta

class BtcStockHandler:
	"""
	This class allows you to access minute-by-minute open, high, low, close (OHLC) 
	Bitcoin stock data from https://www.bitstamp.net. There is a request limit of 
	8000 requests / 10 minutes, which should be plenty to work with. To help you be
	mindful of the request limit, the verbose option is set to True by default for 
	all methods, which shows the timestamps at which requests were made. There are
	two ways to use this class:

		1. For obtaining historical minutely data (set `realtime_mode` to False)

		2. For real time data (set `realtime_mode` to True)

	Option 1 is recommended if you expect to spend more time analyzing past trends.
	Option 2 is recommended if you want to use live updates from the API. If you use
	option 2, then after one minute, there should be no more loss in recent data. Also,
	instantiating a DataStore object will immediately spawn a thread that caches prices
	at the beginning of each minute. The thread is shared between ALL instances of the 
	class and it modifies a cache that is also shared between ALL instances. If the thread
	is turned on from one instance, the updates to the cache will be seen by all instances. 
	Similarly, turning off the thread from one instance will cause it to stop for all 
	instances. 
	"""

	_price_data = None
	_btc_logger = None
	_stop_event = None
	_is_updated = None

	def __init__(self, realtime_mode, cache_size):
		self.base_url = 'https://www.bitstamp.net/api/v2/ohlc/btcusd/'
		if cache_size < 2: raise ValueError("Cache size must be greater than 1.")
		if realtime_mode and BtcStockHandler._btc_logger is None:
			BtcStockHandler._price_data = collections.deque(maxlen=cache_size)
			BtcStockHandler._stop_event = threading.Event()
			BtcStockHandler._is_updated = threading.Event()
			BtcStockHandler._btc_logger = threading.Thread(
				target=self._logdata,
				args=()
			)
			BtcStockHandler._btc_logger.start()

	def _dfcache(self):
		BtcStockHandler._is_updated.wait()
		return pd.concat(list(BtcStockHandler._price_data), ignore_index=True)

	def _recover(self, price_df, start, final):
		dates = pd.date_range(
			start = start.strftime('%Y-%m-%d %H:%M:%S'), 
			end   = final.strftime('%Y-%m-%d %H:%M:%S'),
			freq  = 'T'
		)

		# 1. Re-index the prices by date so that missing dates have NaN
		# 2. Fill in the missing dates using the cache
		# 3. Transform the result back to the original format
		cache = self._dfcache()
		recov = price_df.set_index('timestamp')\
						.reindex(dates)\
						.rename_axis(index=['timestamp'])\
						.fillna(cache.set_index('timestamp'))\
						.reset_index()

		return recov
		
	def _logdata(self):
		BtcStockHandler._price_data.append(self.get_current())
		time.sleep(60 - datetime.now().second)
		while not BtcStockHandler._stop_event.is_set():
			dtnow = datetime.now()
			dtnow = dtnow.replace(second=0, microsecond=0)
			price = self.get_current()
			if price['timestamp'].iloc[0] == dtnow:
				BtcStockHandler._price_data.append(price)
			else:
				time.sleep(2)
				price = self.get_current()
				BtcStockHandler._price_data.append(price)
			BtcStockHandler._is_updated.set()

			# We need to clear the flag early here. If we clear the flag at the 
			# beginning of each iteration, it's possible that we could have a 
			# request come in (1) while the flag is being updated or (2) at the same
			# time we reset the flag. In either case, the flag will still be set to its
			# old value from the previous iteration and the request will think that
			# the cache is updated for the current minute even though it only has
			# data from the previous iteration. This results in stale prices being
			# returned.
			time.sleep(55 - datetime.now().second)
			BtcStockHandler._is_updated.clear()
			time.sleep(60 - datetime.now().second)

	def _request(self, step, limit, start=None, end=None):
		start = '&start=' + str(start) if start is not None else ''
		end   = '&end='   + str(end)   if end   is not None else ''
		rqst = requests.get(self.base_url + '?limit={}&step={}{}{}'.format(limit, step, start, end))
		return json.loads(rqst.content)

	def _timefmt(self, data_dict):
		data = pd.DataFrame(data_dict['data']['ohlc'])
		if 'timestamp' in data.columns:
			data['timestamp'] = pd.to_datetime(data['timestamp'].astype(int).apply(datetime.fromtimestamp))
		return data

	def _collect(self, start, final, delay, verbose):
		
		data_frame = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
		
		# For realtime requests, we sometimes need to wait a bit for the API to update data.
		dif = (datetime.now() - start).total_seconds()
		tol = 3
		if dif <= tol:
			time.sleep(tol - dif)

		if start == final:
			unix_start = int(time.mktime(start.timetuple()))
			data_reqst = self._request(60, 1, unix_start, unix_start)
			data_frame = self._timefmt(data_reqst)
			time_stamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
			start = datetime.fromtimestamp(int(data_reqst['data']['ohlc'][-1]['timestamp']))
			if verbose: print("({}): Collected prices up to {}".format(time_stamp, start))
			return data_frame

		# The API may have missing data, so we can't iterate in evenly-sized increments.
		# Instead, we have to manually check the timestamp of the latest data point to 
		# see if we still need to iterate. Note that if we were to use != or <= in the
		# loop condition we would have an infinite loop. This is why we need to handle
		# the special case when start == final above.
		ohlcv_list = []
		while start < final:
			unix_start = int(time.mktime(start.timetuple()))
			unix_final = unix_start + 59940 # get the start minute plus 999 more minutes (1000 total)
			data_reqst = self._request(60, 1000, unix_start, unix_final)
			ohlcv_list.append(self._timefmt(data_reqst))				
			time_stamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
			start = datetime.fromtimestamp(int(data_reqst['data']['ohlc'][-1]['timestamp']))
			if verbose: print("({}): Collected prices up to {}".format(time_stamp, start))
			start += timedelta(minutes=1) # We want to get prices at the minute after the one we just collected
			time.sleep(delay)

		return pd.concat(ohlcv_list, ignore_index=True)

	def _datefmt(self, start, final):
		start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
		dtnow = datetime.now()
		if final is None:
			final = dtnow
		else:
			final = datetime.strptime(final, "%Y-%m-%d %H:%M:%S")
			final = min(dtnow, final)
		start = start.replace(second=0, microsecond=0)
		final = final.replace(second=0, microsecond=0)
		if start > dtnow: raise ValueError("Start date must not be in the future.")
		if start > final: raise ValueError("Start date must be earlier than final date.")
		return start, final

	def _extract_year(self, date_str):
		return date_str[:date_str.find('-')]

	def _extract_month(self, date_str):
		return date_str[date_str.find('-')+1 : date_str.rfind('-')]

	def _create_folders(self, start, end):
		for d in datastore.DataStore._daterange(start, end):
		    year = 'stocks' + str(d.year)
		    mnth = self._extract_month(d.strftime('%Y-%m-%d'))
		    path = os.path.join(year, mnth)
		    if not os.path.exists(year):
		        os.mkdir(year)
		        os.mkdir(path)
		    else:
		        if not os.path.exists(path): os.mkdir(path)

	def get_cache(self, refresh=False, delay=2, verbose=True):
		"""
		Get all data currently in the cache.

		Parameter(s):
		-------------
			refresh : bool
				If True, sends a request to Bitstamp to get prices at the end
				of each minute. The second to last price may not be updated 
				because of the Bitstamp bug, but it WILL appear in the output. 
				If False, return prices as they were recorded at the beginning 
				of each minute.

			delay : int
				Bitstamp has a request limit of 8000 requests every 10 minutes. 
				Feel free to tweak this for heavy loads so that they don't ban 
				your IP address. This only has an effect if `refresh` is True.

			verbose : bool
				If True, prints the progress after each iteration. Only has an 
				effect if `refresh` is True.

		Returns:
		--------
			A pandas dataframe containing the contents of the cache.
		"""
		if BtcStockHandler._price_data is not None and len(BtcStockHandler._price_data) > 0:
			cache = self._dfcache()
			if refresh:
				start = cache['timestamp'].iloc[ 0].strftime('%Y-%m-%d %H:%M:%S')
				final = cache['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
				return self.get_by_range(start, final)
			return cache

	def is_updated(self):
		"""
		Returns True if the cache has data for the curent minute and False 
		otherwise.
		"""
		if self.is_caching(): return BtcStockHandler._is_updated.is_set()

	def is_caching(self):
		"""
		Returns True if caching is active and False otherwise.
		"""
		return BtcStockHandler._stop_event is not None and not BtcStockHandler._stop_event.is_set()

	def run_caching(self):
		"""
		Turns price caching on. Only has an effect if realtime mode is on.
		If there are multiple instances of this class, turns on price caching
		for ALL of them. This will clear any data currently inside the cache.
		"""
		if not self.is_caching():
			BtcStockHandler._price_data.clear()
			BtcStockHandler._stop_event.clear()
			BtcStockHandler._btc_logger = threading.Thread(
				target=self._logdata,
				args=()
			)
			BtcStockHandler._btc_logger.start()

	def end_caching(self, wait=True):
		"""
		Turns price caching off. Only has an effect if realtime mode is on.
		If there are multiple instances of this class, turns off price caching
		for ALL of them.

		Parameter(s):
		-------------
			wait : bool
				If True, wait for the thread to fully finish running. If
				False, return control immediately and let the thread exit
				on its own.
		"""
		if self.is_caching():
			BtcStockHandler._stop_event.set()
			if wait: BtcStockHandler._btc_logger.join()

	def get_daily(self, start, final=None, delay=0, fname=None, verbose=True):
		"""
		Get all daily OHLCV data from `start` to `final` as a pandas dataframe.
		
		Parameter(s):
		-------------
			start : string
				The start date (inclusive). Must be in the format YYYY-mm-dd.

			final : string or None
				The end date (exclusive). Must be in the format YYYY-mm-dd. If
				None (the default), get all data from `start` to now.

			delay : int
				Bitstamp has a request limit of 8000 requests every 10 minutes. Feel 
				free to tweak this for heavy loads so that they don't ban your IP 
				address.

			verbose : bool
				If True, prints the progress after each iteration.

		Returns:
		--------
			A pandas dataframe with open, high, low, close, volume data. If start and
			final are equal or start is in the future, returns None.
		"""
		if fname is not None and os.path.exists(fname): return

		start = datetime.strptime(start, "%Y-%m-%d")
		dtnow = datetime.now().replace(minute=0, second=0, microsecond=0)
		if final is not None:
			final = datetime.strptime(final, "%Y-%m-%d")
			if final > dtnow:
				final = dtnow
		else:
			final = dtnow

		if start >= final.replace(hour=0): return

		ohlcv_list = []
		resolution = 86400
		while start <= final:
			days_betwn = (final - start).days
			data_limit = min(days_betwn, 1000)
			unix_start = int(time.mktime(start.timetuple()))
			if days_betwn > 1000:
				unix_final = int(time.mktime((start + timedelta(days=999)).timetuple()))
			else:
				unix_final = int(time.mktime(final.timetuple()))
			data_reqst = self._request(resolution, data_limit, unix_start, unix_final)
			ohlcv_list.append(self._timefmt(data_reqst))				
			time_stamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
			start = datetime.fromtimestamp(int(data_reqst['data']['ohlc'][-1]['timestamp']))
			if verbose: print("({}): Collected prices up to {}".format(time_stamp, start))
			start += timedelta(days=1) # We want to get prices on the day after the one we just collected
			time.sleep(delay)

		ohlcv_frame = pd.concat(ohlcv_list, ignore_index=True)
		float_colms = ['open', 'high', 'low', 'close', 'volume']
		ohlcv_frame[float_colms] = ohlcv_frame[float_colms].astype(float)
		if fname is None:
			return ohlcv_frame
		else:
			ohlcv_frame.to_csv(fname, index=False)

	def mass_download(self, start, final, delay=5, verbose=True):
		"""
		Get all minute-by-minute data from `start` to `final` and store all
		files away neatly.
		
		Parameter(s):
		-------------
			start : string
				The start date (inclusive). Must be in the format YYYY-mm-dd.

			final : string
				The end date (exclusive). Must be in the format YYYY-mm-dd.

			delay : int
				Bitstamp has a request limit of 8000 requests every 10 minutes. Feel 
				free to tweak this for heavy loads so that they don't ban your IP 
				address.

			verbose : bool
				If True, prints the progress after each iteration.

		Returns:
		--------
			None. Data is stored in CSV files.
		"""
		start = datetime.strptime(start, '%Y-%m-%d')
		final = datetime.strptime(final, '%Y-%m-%d')
		self._create_folders(start, final)
		for d in datastore.DataStore._daterange(start, final):
			start = d.replace(hour=00, minute=00, second=00).strftime('%Y-%m-%d %H:%M:%S')
			final = d.replace(hour=23, minute=59, second=59).strftime('%Y-%m-%d %H:%M:%S')
			fdate = d.strftime('%Y-%m-%d')
			fname = 'stocks{}/'.format(self._extract_year(fdate)) + self._extract_month(fdate) + '/' + fdate + '-stocks.csv'
			self.get_by_range(start, final, delay, fname, verbose)

	def get_current(self):
		"""
		Get the current price of Bitcoin.

		Returns:
		--------
			A pandas dataframe with a single row containing the most recent 
			ohlcv data for Bitcoin.
		"""
		unix_now = int(time.mktime(datetime.now().timetuple()))
		return self._timefmt(self._request(60, 1, unix_now, unix_now))

	def get_by_day(self, date, delay=0, fname=None, verbose=True):
		"""
		Get the next 24 hour's worth of stock data starting from `date`.
		
		Parameter(s):
		-------------
			date : string
				The day to get data. Must be in the format YYYY-mm-dd HH:MM:SS.

			delay : int
				Bitstamp has a request limit of 8000 requests every 10 minutes. Feel 
				free to tweak this for heavy loads so that they don't ban your IP 
				address.

			fname : None or string
				If specified, results are stored in a csv file. Otherwise, data is
				returned in a dataframe.

			verbose : bool
				If True, prints the progress after each iteration.

		Returns:
		--------
			If `fname` is None, returns a pandas dataframe. Otherwise, data is stored in
			a CSV file.

		Notes:
		-----
			If caching is active, then data recovery for the Bitstamp bug will 
			automatically occur.
		"""
		start = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
		final = (start + timedelta(days=1))
		start = start.strftime("%Y-%m-%d %H:%M:%S")
		final = final.strftime("%Y-%m-%d %H:%M:%S")
		return self.get_by_range(start, final, delay, fname, verbose).iloc[:1440]

	def get_by_month(self, date, delay=0, fname=None, verbose=True):
		"""
		Get all minute-by-minute data for a particular month.
		
		Parameter(s):
		-------------
			date : string
				The month to get data. Must be in the format YYYY-mm.

			delay : int
				Bitstamp has a request limit of 8000 requests every 10 minutes. Feel 
				free to tweak this for heavy loads so that they don't ban your IP 
				address.

			fname : None or string
				If specified, results are stored in a csv file. Otherwise, data is
				returned in a dataframe.

			verbose : bool
				If True, prints the progress after each iteration.

		Returns:
		--------
			If `fname` is None, returns a pandas dataframe. Otherwise, data is stored in
			a CSV file.

		Notes:
		-----
			If caching is active, then data recovery for the Bitstamp bug will 
			automatically occur.
		"""
		start = datetime.strptime(date, '%Y-%m')
		final = calendar.monthrange(start.year, start.month)[1]
		final = start.replace(day=final , hour=23, minute=59, second=59).strftime('%Y-%m-%d %H:%M:%S')
		start = start.replace(day=1		, hour=00, minute=00, second=00).strftime('%Y-%m-%d %H:%M:%S')
		return self.get_by_range(start, final, delay, fname, verbose)

	def get_by_range(self, start, final=None, delay=2, fname=None, verbose=True):
		"""
		Get all minute-by-minute data from `start` to `final`. By default, data is
		stored in a dataframe.
		
		Parameter(s):
		-------------
			start : string
				The start date (inclusive). Must be in the format YYYY-mm-dd HH:MM:SS.

			final : string
				The end date (inclusive). Must be in the format YYYY-mm-dd HH:MM:SS. If
				None (the default), get all data from `start` to now.

			delay : int
				Bitstamp has a request limit of 8000 requests every 10 minutes. Feel 
				free to tweak this for heavy loads so that they don't ban your IP 
				address.

			fname : None or string
				If specified, results are stored in a csv file. Otherwise, data is
				returned in a dataframe.

			verbose : bool
				If True, prints the progress after each iteration.

		Returns:
		--------
			If `fname` is None, returns a pandas dataframe. Otherwise, data is stored in
			a CSV file.

		Notes:
		-----
			If caching is active, then data recovery for the Bitstamp bug will 
			automatically occur.
		"""
		if fname is not None and os.path.exists(fname): return
		start, final = self._datefmt(start, final)
		ohlcv_frame = self._collect(start, final, delay, verbose)
		ohlcv_frame = ohlcv_frame[pd.to_datetime(ohlcv_frame['timestamp']) <= final]

		# Realtime recovery
		if self.is_caching():
			ohlcv_frame = self._recover(ohlcv_frame, start, final)

		float_cols = ['open', 'high', 'low', 'close', 'volume']
		ohlcv_frame[float_cols] = ohlcv_frame[float_cols].astype(float)
		if fname is None:
			return ohlcv_frame
		else:
			ohlcv_frame.to_csv(fname, index=False)

	def refresh(self, data_frame, compress=False, delay=2, verbose=True):
		"""
		Gathers any new stock prices from the latest date in `data_frame` to now.

		Parameter(s):
		-------------
			data_frame : pandas dataframe
				The dataframe returned on a previous call to `get_by_*()`. The most
				recent date in `data_frame` must be within 48 hours of the current
				time. For larger requests, use the other functions.

			compress : bool
				If True, returns only the new data points instead of the
				combination of original and new data points. 

			delay : int
				Bitstamp has a request limit of 8000 requests every 10 minutes. 
				Feel free to tweak this for heavy loads so that they don't ban
				your IP address.

			verbose : bool
				If True, prints the progress after each iteration.

		Returns:
		--------
			A new dataframe with the original data plus any new data. If `compress`
			is True, only returns a dataframe with the new data points.

		Notes:
		-----
			If caching is active, then data recovery for the Bitstamp bug will 
			automatically occur.
		"""
		start = pd.to_datetime(data_frame['timestamp']).max()
		if (datetime.now() - start).total_seconds() > (86400 * 2) + 1:
			raise ValueError("Time range to large.")
		if self.is_caching():
			cache = self.get_cache(refresh=True, delay=delay, verbose=verbose)
			if compress: return cache[cache['timestamp'] >= start]
			dfcpy = data_frame[data_frame['timestamp'] < cache['timestamp'].values[0]]
			return pd.concat([dfcpy, cache], ignore_index=True)
		else:
			extra = self.get_by_range(str(start), None, delay, None, verbose)
			if compress: return extra
			dfcpy = data_frame[data_frame['timestamp'] < start]
			return pd.concat([data_frame, extra], ignore_index=True)

	def transactions(self, interval, verbose=True):
		"""
		Get all Bitcoin stock transactions in the past minute, hour, or day.

		Parameter(s):
		-------------
			interval : string
				One of 'minute', 'hour', or 'day'.

			verbose : bool
				If True, prints the time at which the request was made.

		Returns:
		--------
			A new dataframe with the following columns:

				date	: Unix timestamp date and time.
				tid	: Transaction ID.
				price 	: BTC price.
				amount 	: BTC amount.
				type 	: 0 (buy) or 1 (sell)
		"""
		interval = interval.lower()
		assert interval in ['minute', 'hour', 'day'], 'Invalid interval specified.'
		reqst = requests.get('https://www.bitstamp.net/api/v2/transactions/btcusd/?time={}'.format(interval))
		time_stamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
		if verbose: print("({}): Collected transaction data!".format(time_stamp))
		trxns = pd.DataFrame(json.loads(reqst.content))
		if 'date' in trxns.columns:
			trxns['date'] = pd.to_datetime(trxns['date'].astype(int).apply(datetime.fromtimestamp))
		return trxns
