import pandas as pd
import numpy as np

class Spliter(object):
    def __init__(self, file_name, number_of_files = 10, split=True):
        self.master_file_name = file_name # 'WebData_01.csv'
        self.all_data = pd.read_csv(self.master_file_name)
        self.is_split = split
        if self.is_split:
            self.split(number_of_files)
        else:
            self.build_map()

    def map_categories(self, by_column = []):
        if by_column == [] :
            m_list = sorted(list(set(self.all_categories)))
        else:
            m_list = sorted(list(set(by_column)))
        category_dict = {}
        for index, category in enumerate(m_list):
            category_dict[category] = index
        return category_dict

    def set_y_values(self):
        y_column = list()
        for category in self.all_categories:
            y_column.append(self.category_mapping[category])
        self.all_data['y'] = y_column

    def df_split_files(self, number_of_files):
        df_split = np.array_split(self.all_data, number_of_files)
        for index, df in enumerate(df_split):
            # df.reset_index(drop=True)
            df.to_csv('Data_Split_'+str(index)+'.csv')

    def split(self, number_of_files, map_by = 'Category'):
        self.all_categories = self.all_data[map_by]
        self.category_mapping = self.map_categories()
        self.set_y_values()
        self.df_split_files(number_of_files)

    def build_map(self):
        self.mapping_data = pd.read_csv("Master_Data_1.csv")
        map_from_by_column_name = str(self.all_data.columns[1])
        map_to_by_column_name = str(self.all_data.columns[0])

        start_map = {}
        self.all_Primary_categories = self.all_data[map_to_by_column_name]
        self.all_Secondary_categories = self.all_data[map_from_by_column_name]

        self.all_Primary_mapping = self.map_categories(list(self.all_Primary_categories))
        self.all_Secondary_mapping = self.map_categories(list(self.all_Secondary_categories))

        for Primary, Secondary in zip(list(self.all_Primary_categories), list(self.all_Secondary_categories)):
            start_map[Secondary] = Primary

        main_map = {}
        # opposite sides key = Secondary category, value  = Primary category
        for start_key, start_value in start_map.items():
            main_map[(str(self.all_Secondary_mapping[start_key]))] = str(self.all_Primary_mapping[start_value])

        y_list = self.mapping_data[self.mapping_data.columns[-1]]
        y_tag = []
        for y in y_list:
            y_tag.append(main_map[str(y)])


if __name__ == "__main__":
    inst = Spliter(file_name = 'WebData_02.csv', split=False)
