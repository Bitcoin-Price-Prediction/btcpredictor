import pyrebase
import os

class Database:

	fb = None
	rt = None
	db = None

	@staticmethod
	def connect():

		config = {
		    "apiKey"            : os.getenv('FB_apiKey'),
		    "authDomain"        : os.getenv('FB_authDomain'),
		    "databaseURL"       : os.getenv('FB_databaseURL'),
		    "projectId"         : os.getenv('FB_projectId'),
		    "storageBucket"     : os.getenv('FB_storageBucket'),
		    "messagingSenderId" : os.getenv('FB_messagingSenderId'),
		    "appId"             : os.getenv('FB_appId'),
		    "measurementId"     : os.getenv('FB_measurementId')
		}

		print('Connecting to databases... ', end='')
		Database.fb = pyrebase.initialize_app(config)
		Database.rt = Database.fb.database()
		Database.db = Database.fb.storage()
		print('done!')