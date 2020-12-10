from .reporter import *
from .database import *
from .helpers  import *
import gzip
import json
import os

class AbstractLogger:

	@staticmethod
	def store(api_name, progress, offset=None):
		try:
			data = Database.rt.child(api_name).get().val()
			Database.rt.child(api_name).remove()            
			date = localtime() if offset is None else localtime() + offset
			name = date.strftime("%Y-%m-%d") + "-{}.json".format(api_name)
			path = "{}/{}/{}/{}".format(api_name, date.year, date.month, name)
			with gzip.GzipFile(name, "w") as outfile:
				outfile.write(json.dumps({ 'data' : data }).encode('utf-8'))
			Database.db.child(path).put(name)
			os.remove(name)
			Reporter.append('({}) {} log count: {}'.format(localtime(fmt=True), api_name, len(data) if data else 0))
			if progress: print("Stored {} data!".format(api_name))
		except Exception as e:
			Reporter.send_urgent(api_name, "storing", repr(e))

	@staticmethod
	def log():
		raise NotImplementedError()