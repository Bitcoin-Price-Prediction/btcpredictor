from .reporter import *
from .database import *
from .helpers  import *
from .logger   import *
import requests

class TckrLogger(AbstractLogger):

    base_url = 'https://www.bitstamp.net/api/v2/ticker/btcusd/'
    logrname = 'Stocks'
    progress = True

    @staticmethod
    def store():
        this = TckrLogger

        # When transactions were introduced, a storing delay for stocks appeared. To account 
        # for the delay, we rewind by 1 hour, which should ensure the files are named properly.
        super(this, this).store(this.logrname, this.progress, offset = -timedelta(hours=1))

    @staticmethod
    def log():
        try:
            ltime = localtime(fmt=True)
            reqst = requests.get(TckrLogger.base_url)
            items = json.loads(reqst.content)
            items["LAtime"] = ltime
            items = Database.rt.child(TckrLogger.logrname).push(items)
            if TckrLogger.progress: print("Logged {} data: {}".format(TckrLogger.logrname, items))
        except Exception as e:
            Reporter.send_urgent(TckrLogger.logrname, "logging", repr(e))