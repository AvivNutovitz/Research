import pymongo
import pandas as pd

class MongoHandler():
    # here the database 'report_engine' and the collection test_suites are hard coded
    def __init__(self, collection_name):
        self.db_name = 'urlsDB'
        self.connection_string = 'mongodb://132.66.196.188:27017'
        self.database = self._db_connection(self.db_name)
        self.collection = self.database[collection_name]
        self.main_df = pd.DataFrame()

    def save_json_data(self, json_data):
        try:
            self.collection.insert_one(json_data)
        except:
            pass

    def _db_connection(self, db_name):
        # connection_string = "mongodb://pt-lt0302"
        connection = pymongo.MongoClient()#self.connection_string)
        # connection = pymongo.MongoClient()
        return connection[db_name]

    def build_single_json(self):

        # get all data from the collection
        cursor = self.collection.find({})

        for document in cursor:
            temp_df = pd.DataFrame(data=document, index=[document['index']])
            self.main_df = self.main_df.append(temp_df)

        self.main_df.to_csv('Master_Data.csv')

# if __name__ == "__main__":
#     mh = MongoHandler('urlsCollections')
#     mh.build_single_json()
