from websites_data.Aggregator import FileAggregator
from websites_data.Parser import WebParser
from websites_data.Installer import Installer
from websites_data.MongoHandler import MongoHandler
from websites_data.Moduler import Moduler
from websites_data import Parser
import pandas as pd
import fnmatch
import os

class Runner(object):
    def __init__(self, collection_name='urlsCollections', file_name='', mode='train', user_test_url = ''):
        # color dict
        self.color_dict = {
    'Red_Saturated': (243, 60, 81),
    'Red_Light': (255, 160, 163),
    'Red_Muted': (200, 103, 107),
    'Red_Dark': (140, 43, 55),
    'Orange_Saturated': (255, 157, 47),
    'Orange_Light': (255, 200, 163),
    'Orange_Muted': (210, 144, 108),
    'Orange_Dark': (137, 77, 43),
    'Yellow_Saturated': (255, 241, 48),
    'Yellow_Light': (255, 243, 153),
    'Yellow_Muted': (215, 185, 102),
    'Yellow_Dark': (143, 116, 40),
    'Chartreuse_Saturated': (200, 229, 61),
    'Chartreuse_Light': (226, 238, 154),
    'Chartreuse_Muted': (166, 181, 106),
    'Chartreuse_Dark': (109, 126, 53),
    'Green_Saturated': (0, 200, 130),
    'Green_Light': (150, 225, 185),
    'Green_Muted': (101, 174, 137),
    'Green_Dark': (29, 112, 79),
    'Cyan_Saturated': (0, 209, 205),
    'Cyan_Light': (155, 229, 228),
    'Cyan_Muted': (96, 172, 172),
    'Cyan_Dark': (33, 115, 117),
    'Blue_Saturated': (34, 169, 239),
    'Blue_Light': (161, 207, 251),
    'Blue_Muted': (104, 151, 192),
    'Blue_Dark': (41, 97, 137),
    'Purple_Saturated': (174, 71, 217),
    'Purple_Light': (217, 169, 250),
    'Purple_Muted': (161, 113, 189),
    'Purple_Dark': (108, 55, 132),
    'Black_Achromatic': (10, 10, 11),
    'DarkGray_Achromatic': (101, 97, 103),
    'MedGray_Achromatic': (157, 151, 160),
    'LightGray_Achromatic': (214, 207, 218),
    'White_Achromatic': (255, 255, 255),
}
        # the best practise is to send one csv file to one runner
        self.df = pd.DataFrame()
        # collection name
        self.collection_name = collection_name
        # get running mode
        self.mode = mode
        # get specific url to test
        self.specific_url = user_test_url
        # find file name to run
        if file_name != '':
            # the user provided file name, user file must have an index
            self.df = self.read_data_frame(file_name)
        else:
            for file in os.listdir('.'):
                if fnmatch.fnmatch(file, '*.csv'):
                    # create the data from the file
                    self.df = self.read_data_frame(file)

    def run(self):
        if self.mode == 'train':
            # connect to the DB
            try:
                # connect to MongoDB, to the specific collection
                mh = MongoHandler(self.collection_name)

            except Exception as e:
                print(str(e))
                raise Exception

            for index, (url, y) in enumerate(zip(self.df['URL'], self.df['y'])):
                try:
                    json_of_file, file_name = self.running_block(url=url, index=index, y=y)
                    # send the data to the DB
                    mh.save_json_data(json_of_file)
                    try:
                        # remove the local file is exist
                        os.remove(file_name)
                    except Exception as e:
                        print(str(e))

                except Exception as e:
                    print(str(e))

        elif self.mode == 'test':

            model = Moduler(mode='test')

            # user test a single url
            if self.df.empty and self.specific_url != '':
                try:
                    json_of_file, file_name = self.running_block(url=self.specific_url, index=1)

                    try:
                        # remove the local file is exist
                        os.remove(file_name)
                    except Exception as e:
                        print(str(e))

                    model.test_model(json_of_file)

                except Exception as e:
                    print(str(e))

            else:
                # user send many websites to test
                for index, url in enumerate(self.df['URL']):
                    try:
                        json_of_file, file_name = self.running_block(url=url, index=index)

                        try:
                            # remove the local file is exist
                            os.remove(file_name)
                        except Exception as e:
                            print(str(e))

                        model.test_model(json_of_file)

                    except Exception as e:
                        print(str(e))

        else:
            print("there is no such mode called {}".format(self.mode))

    def read_data_frame(self, file_name):
        try:
            df = pd.read_csv(file_name, index_col=0)
        except:
            try:
                df = pd.read_csv(file_name, encoding = 'ISO-8859-1', index_col=0)
            except:
                raise Exception
        return df

    def running_block(self, url, index, y=None):

        # run the web parser - create local csv file from url
        WebParser(Parser.readable_url(url, True), self.color_dict)
        # build file name for the csv file
        file_name = Parser.readable_url(url, False)
        # run the json aggregator
        json_of_file = FileAggregator(file_name, self.color_dict).get_aggregator_json()
        # adding information to the json
        json_of_file['index'] = index
        json_of_file['URL'] = Parser.readable_url(url, True)
        if self.mode == 'train':
            # only for training
            json_of_file['y'] = y
        return json_of_file, file_name

if __name__ == "__main__":
    # inst = Installer()
    # inst.install()
    runner = Runner()
    runner.run()