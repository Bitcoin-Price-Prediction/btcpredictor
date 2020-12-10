from datetime import timedelta, datetime
from calendar import monthrange
import pytz

def seconds_until_next_month():
    today = datetime.utcnow()
    ndays = monthrange(today.year, today.month)[1]
    nextm = today.replace(day=1) + timedelta(days=ndays)
    nextm = datetime.combine(nextm, datetime.min.time())
    return (nextm - datetime.utcnow()).total_seconds()

def localtime(localtz="America/Los_Angeles", fmt=False):
	time = datetime.now(pytz.timezone(localtz))
	if fmt:
		return time.strftime("%Y-%m-%d %H:%M:%S")
	else:
		return time

def ceiltime(dt, delta):
	return dt + (datetime.min - dt) % delta

def next30sec(dt):
	dt = dt.replace(microsecond=0)
	if dt.second < 30:
		return dt.replace(second=30)
	else:
		return dt.replace(minute=dt.minute+1, second=30)