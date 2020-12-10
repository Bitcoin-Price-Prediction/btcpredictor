import datastore.datastore as datastore
import pandas as pd
import shutil
import json
import gzip
import glob
import re
import os

from datetime import datetime, timedelta


class ArchivesHandler:
	"""
	This class allows you to access stock and news data from previous days.
	All methods will NOT do bounds checking: if you input a very large range
	of dates, all dates will be inspected regardless of whether a file exists
	at that date or not. So far, stock data goes back to 2020-10-27 and news 
	data goes back to 2020-10-23.
	"""

	def _datefmt(self, start, final):
		start = datetime.strptime(start, "%Y-%m-%d")
		dtnow = datetime.now()
		if final is None:
			final = dtnow
		else:
			final = datetime.strptime(final, "%Y-%m-%d")
			final = min(dtnow, final)
		final = final.replace(hour=0, minute=0, second=0, microsecond=0)
		if start > final:
			raise ValueError("Start date must be earlier than final date.")
		return start, final

	def _collect(self, api_name, start, final, dirname, verbose):
		start, final = self._datefmt(start, final)
		path = api_name + "/{}/{}/{}"
		if dirname is not None and not os.path.exists(dirname): os.mkdir(dirname)
		for date in datastore.DataStore._daterange(start, final + timedelta(days=1)):
			name = date.strftime("%Y-%m-%d") + "-{}.*".format(api_name)
			if not (glob.glob(name) or (dirname is not None and glob.glob(os.path.join(dirname, name)))):
				name = date.strftime("%Y-%m-%d") + "-{}.json".format(api_name)
				datastore.DataStore._db.child(path.format(date.year, date.month, name)).download(name)
				if os.path.exists(name):
					with gzip.GzipFile(name, 'r') as f:
						file = json.loads(f.read().decode('utf-8'))
					with open(name, 'w') as f:
						json.dump(file['data'], f)
					if dirname: shutil.move(name, dirname)
					if verbose: print('{} was downloaded!'.format(name))
				else:
					if verbose: print('{} was not found.'.format(name))
			else:
				if verbose: print('{} already exists'.format(name))

	def news(self, start, final=None, dirname=None, verbose=True):
		"""
		Parameter(s):
		-------------
			start : string
				The start date (inclusive). Must be in the format YYYY-mm-dd.

			final : string
				The end date (inclusive). Must be in the format YYYY-mm-dd.

			dirname : None or string
				The name of the output directory to store data. If unspecified, 
				all files are downloaded in the current directory.

			verbose : bool
				If True, print download progress.

		Returns:
		--------
			None. All files in range are downloaded to the specified directory as CSV files.
	    """
		self._collect("News", start, final, dirname, verbose)
		if verbose: print('Converting from JSON to CSV...\t', end='')
		for fname in os.listdir(dirname):
			if re.match(r'^\d{4}-\d{2}-\d{2}-News\.json$', fname):
				path = fname if dirname is None else os.path.join(dirname, fname)
				with open(path, 'r') as f: news = json.load(f)
				data = pd.DataFrame()
				for item in news.values():
				    temp = pd.DataFrame(item['articles'])
				    temp['LAtime'] = item['LAtime']
				    data = pd.concat([data, temp], ignore_index=True)
				data = data.sort_values('datetime')
				data.to_csv(path.replace('.json', '.csv'), index=False)
				os.remove(path)
		if verbose: print('done!')

	def stocks(self, start, final=None, dirname=None, verbose=True):
		"""
		Parameter(s):
		-------------
			start : string
				The start date (inclusive). Must be in the format YYYY-mm-dd.

			final : string
				The end date (inclusive). Must be in the format YYYY-mm-dd.

			dirname : None or string
				The name of the output directory to store data. If unspecified, 
				all files are downloaded in the current directory.

			verbose : bool
				If True, print download progress.

		Returns:
		--------
			None. All files in range are downloaded to the specified directory as CSV files.
	    """
		self._collect("Stocks", start, final, dirname, verbose)
		if verbose: print('Converting from JSON to CSV...\t', end='')
		for fname in os.listdir(dirname):
			if re.match(r'^\d{4}-\d{2}-\d{2}-Stocks\.json$', fname):
				path = fname if dirname is None else os.path.join(dirname, fname)
				with open(path, 'r') as f: stck = json.load(f)
				data = pd.DataFrame(stck.values()).sort_values('timestamp')
				data.to_csv(path.replace('.json', '.csv'), index=False)
				os.remove(path)
		if verbose: print('done!')

	def transactions(self, start, final=None, dirname=None, verbose=True):
		"""
		Parameter(s):
		-------------
			start : string
				The start date (inclusive). Must be in the format YYYY-mm-dd.

			final : string
				The end date (inclusive). Must be in the format YYYY-mm-dd.

			dirname : None or string
				The name of the output directory to store data. If unspecified, 
				all files are downloaded in the current directory.

			verbose : bool
				If True, print download progress.

		Returns:
		--------
			None. All files in range are downloaded to the specified directory as CSV files.
	    """
		self._collect("Transactions", start, final, dirname, verbose)
		if verbose: print('Converting from JSON to CSV...\t', end='')
		for fname in os.listdir(dirname):
			if re.match(r'^\d{4}-\d{2}-\d{2}-Transactions\.json$', fname):
				path = fname if dirname is None else os.path.join(dirname, fname)
				with open(path, 'r') as f: trxn = json.load(f)
				data = pd.DataFrame()
				for item in trxn.values():
				    temp = pd.DataFrame(item['transactions'])
				    temp['LAtime'] = item['LAtime']
				    data = pd.concat([data, temp], ignore_index=True)
				data.to_csv(path.replace('.json', '.csv'), index=False)
				os.remove(path)
		if verbose: print('done!')
