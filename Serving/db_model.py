from pymongo import MongoClient

client = MongoClient()

client = MongoClient('localhost', 27017)

db = client['NFPA']

feedback_collection = db['nfpa_feedback']