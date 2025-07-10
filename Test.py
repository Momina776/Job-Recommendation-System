from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db = client['job_prediction_db']
print(list(db.jobs.find({}, {'title': 1})))