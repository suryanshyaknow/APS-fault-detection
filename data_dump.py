import pymongo
import pandas as pd
import json

# Provide the mongodb client localhost to connect python to mongodb
client = pymongo.MongoClient("mongodb+srv://suryanshyaknow:streamingdata@sensors-streaming-data.kkw0g1z.mongodb.net/test")

DATA_PATH = r"D:\data-science\projects\APS-fault-detection\aps_failure_training_set1.csv"
DATABASE = client["sensors-streaming-data"]
COLLECTION = DATABASE['sensors']


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"Rows and Columns: {df.shape}")

    # Converting the .csv to "dumpable into MongoDB" formate i.e. json
    df.reset_index(drop=True, inplace=True)
    all_records = list(json.loads(df.T.to_json()).values())
    print(all_records[0])

    # Dumping into MongoDb database
    COLLECTION.insert_many(all_records)
