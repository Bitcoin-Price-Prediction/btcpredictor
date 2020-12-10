from util.trxn_logger import *
from util.tckr_logger import *
from util.news_logger import *
from util.database 	  import *
from util.reporter 	  import *

from apscheduler.schedulers.background import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from datetime import timedelta, datetime

# -------------------------------------------------------------------------------------------------------------------------------------------------------
# Summary:		|					|		
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# API Name 		|	Refresh Time 	|	Purpose				|	Link(s) to Docs
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# IEXCloud 		|	~5 minute(s)	|	Real time articles	|	https://iexcloud.io/docs/api/#news
#				|					|	for sentiment		|
#				|					|	analysis. Gathers	|
#				|					|	5 articles every 5	|
#				|					|	minutes.			|
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# Bitstamp		|	~05 seconds		|	Offers very high	|	https://www.bitstamp.net/api/
# 				|	for stocks		|	resolution stock 	|
#				|					|	prices which are 	|
#				| 	~24 hours for	|	pretty accurate.	|
#				|	transactions	|	They also provide	|
#				|					|	transaction data 	|
#				|					|	as well.			|
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# CoinMarketCap	|	------------	|	Bitstamp offers 	|	https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyQuotesLatest
#				|	DISCONTINUED	|	essentially the		|	
#				|	------------	|	same quality of		|	
#				|					|	data at more 		|
#				|					|	frequent intervals.	|
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# Blockchain 	|	------------	|	Decided to get rid	|	https://www.blockchain.com/api/charts_api
#				|	DISCONTINUED	|	of this to save 	|
# 				|	------------	|	space in the 		|
#				|					|	database and get 	|
#				|					|	less timing errors.	|
#				|					|	Plus it's hard to 	|
#				|					|	find historical 	|
#				|					|	blockchain data 	|
#				|					|	that looks like 	|
#				|					|	this.				|
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# NewsAPI 		|	------------	|	IEXCloud offers 	|	https://newsapi.org/docs/endpoints/everything
#				|	DISCONTINUED	|	real time news and 	|
# 				|	------------	|	offers better		|
#				|					|	results.			|
# -------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

	Database.connect()
	time_zone = 'America/Los_Angeles'

	# We don't want to miss out on any ticker data as we store other data, so we increase concurrrency.
	scheduler = BlockingScheduler(
		executors={ 'default' : ThreadPoolExecutor(30) },
		job_defaults={ 'max_instances' : 30 }
	)

	# The last stock and news data points are collected at around 23:59:55, so we'll store the data a bit later.
	store_hr, store_mn, store_se = 23, 59, 58

	# Transactions need to be collected at 23:59:59, so we need to store them a bit later than 23:59:58.
	txlog_hr, txlog_mn, txlog_se = 23, 59, 59
	txstr_hr, txstr_mn, txstr_se = 00, 00, 5
	
	# This should be set to some time after the last store operation. Timing test results indicate that >=3 minutes is ideal.
	email_hr, email_mn, email_se = 00, 3, 00

	# The start time should be slightly ahead of the current time, so the schedulers don't miss it.
	start_delay = 1
	logr_start_time = datetime.now() + timedelta(seconds=start_delay)

	# We'll start the news logger when the next minute % 10 == 0 (e.g. XX:10:00, XX:20:00, ...)
	news_start_time = ceiltime(logr_start_time, timedelta(minutes=5))

	# We'll start the ticker logger when the next second % 5 == 0 (e.g. XX:XX:05, XX:XX:10, ...)
	tckr_start_time = ceiltime(logr_start_time, timedelta(seconds=5))

	scheduler.add_job(
		NewsLogger.log, 
		'interval', 
		minutes=NewsLogger.INTERVAL,
		next_run_time=news_start_time, 
		misfire_grace_time=60
	)
	
	scheduler.add_job(
		TckrLogger.log, 
		'interval', 
		seconds=5, 
		next_run_time=tckr_start_time, 
		misfire_grace_time=3
	)
	
	scheduler.add_job(
		TrxnLogger.log, 
		'cron', 
		hour=txlog_hr, 
		minute=txlog_mn, 
		second=txlog_se, 
		timezone=time_zone,
		misfire_grace_time=3
	)

	# Store data and send an email notification
	scheduler.add_job(NewsLogger.store,	'cron',	hour=store_hr, minute=store_mn, second=store_se, timezone=time_zone, misfire_grace_time=60)
	scheduler.add_job(TckrLogger.store,	'cron', hour=store_hr, minute=store_mn, second=store_se, timezone=time_zone, misfire_grace_time=60)
	scheduler.add_job(TrxnLogger.store, 'cron', hour=txstr_hr, minute=txstr_mn, second=txstr_se, timezone=time_zone, misfire_grace_time=60)
	scheduler.add_job(Reporter.send_report, 'cron', hour=email_hr, minute=email_mn, second=email_se, timezone=time_zone, misfire_grace_time=60)

	print("Starting the scheduler.")
	scheduler.start()