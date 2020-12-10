from .reporter import *
from .database import *
from .helpers  import *
from .logger   import *
import requests

class TrxnLogger(AbstractLogger):

    base_url = 'https://www.bitstamp.net/api/v2/transactions/btcusd/?time=day'
    logrname = 'Transactions'
    progress = True

    @staticmethod
    def store():
        this = TrxnLogger

        # Transactions are collected at 23:59:59 on day X, but they are stored at 00:00:05 on day X + 1.
        # This means that the output file will be named using day X + 1. To avoid this, we need to rewind
        # time back to the previous day, so that the file is named correctly. A 1 hour rewind should be more 
        # than enough to bring us back to day X. What really matters is that the transactions file for day X 
        # is named correctly using day X.
        super(this, this).store(this.logrname, this.progress, offset = -timedelta(hours=1))

    @staticmethod
    def log():
        try:
            ltime = localtime(fmt=True)
            reqst = requests.get(TrxnLogger.base_url)
            items = json.loads(reqst.content)
            items = {
                "transactions"  : items,
                "LAtime"        : ltime
            }
            items = Database.rt.child(TrxnLogger.logrname).push(items)
            if TrxnLogger.progress: print("Logged {} data: {}".format(TrxnLogger.logrname, items))
        except Exception as e:
            Reporter.send_urgent(TrxnLogger.logrname, "logging", repr(e))