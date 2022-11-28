import pymongo

# Provide the mongodb client localhost to connect python to mongodb
client = pymongo.MongoClient("mongodb+srv://suryanshyaknow:streamingdata@sensors-streaming-data.kkw0g1z.mongodb.net/test")

# Database Name
database = client["sensors-streaming-data"]

# Collection Name
collection = database['Author']

# Author Info
details = {
    'authorName': 'Suryansh Grover',
    'authorMail': 'suryanshgrover1999@gmail.com',
}

# Insert `details` in the collection
rec = collection.insert_one(details)

# Verify all the records present in the collection at present
all_recs = collection.find()

# printing those records
print(all_recs)
