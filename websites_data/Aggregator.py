import pandas as pd
import numpy as np
import colorgram
from PIL import Image
from websites_data import Parser

class FileAggregator(object):
    def __init__(self, file_name, color_dict):

        self.df = self.read_data_frame(file_name)
        self.color_dict = color_dict
        self.general_colors = self.get_general_colors_and_shades(0)
        self.general_shades = self.get_general_colors_and_shades(1)
        self.aggregator_json = {}
        self.get_totals(self.aggregator_json)
        # print(self.aggregator_json)

    def read_data_frame(self, file_name):
        try:
            df = pd.read_csv(file_name)
        except:
            try:
                df = pd.read_csv(file_name, encoding = 'ISO-8859-1')
            except:
                raise Exception
        return df

    def get_aggregator_json(self):
        return self.aggregator_json

    def get_totals(self, aggregator_json):
        aggregator_json['number_of_elements_from_file'] = float(self.get_number_of_elements_from_file(self.df))
        aggregator_json['number_of_different_tag_names'] = float(self.get_number_of_different_tag_names_from_file(self.df))
        aggregator_json['number_of_all_color_elements'] = float(self.get_number_of_all_color_elements(self.df, 'color_closest_str'))
        aggregator_json['number_of_all_background_color_elements'] = float(self.get_number_of_all_color_elements(self.df, 'background_color_closest_str'))
        aggregator_json['number_of_text_elements'] = float(self.get_number_of_not_NaN_elements_from_file_by_parameter(self.df, 'text_length'))
        aggregator_json['number_of_area_elements'] = float(self.get_number_of_not_NaN_elements_from_file_by_parameter(self.df, 'element_area'))

        for tag_name in ['img', 'a', 'div', 'span', 'h1', 'h2', 'h3']:
            aggregator_json['number_of_'+tag_name] = float(self.get_number_of_elements_from_file_by_parameter(self.df, tag_name))
        for position in ['absolute', 'relative']:
            aggregator_json['number_of_elements_with_'+position+'_position'] = float(self.get_number_of_elements_from_file_by_parameter(self.df, position, column_name='position'))
        for element in ['text_length', 'number_of_words', 'text_total_rate', 'element_area']:
            aggregator_json['total_'+element+'_elements'] = self.get_totals_from_file_by_parameter(self.df, element)
        for column_name in ['background_color_closest_str', 'color_closest_str', 'color_closest_str_with_HLS', 'background_color_closest_str_with_HLS']:
            for name, key in zip(["specific_color", "general_color", "shade_key"], [1, 2, 3]):
                aggregator_json["entropy_of_"+name+"_from_column_"+column_name] = float(self.get_entropy_by_column_name_and_key(self.df, column_name, key))

            for specific_color_key, color_value in self.color_dict.items():
                aggregator_json['number_of_elements_of_specific_color_' + specific_color_key + '_from_' + column_name] = float(self.get_number_or_percentage_of_color_elements(self.df, specific_color_key, column_name, True, 1))
                aggregator_json['percentage_of_elements_of_specific_color_' + specific_color_key + '_from_' + column_name] = float(self.get_number_or_percentage_of_color_elements(self.df, specific_color_key, column_name, False, 1))

            for general_color_key in self.general_colors:
                aggregator_json['number_of_elements_of_general_color_' + general_color_key + '_from_' + column_name] = float(self.get_number_or_percentage_of_color_elements(self.df, general_color_key, column_name, True, 2))
                aggregator_json['percentage_of_elements_of_general_color_' + general_color_key + '_from_' + column_name] = float(self.get_number_or_percentage_of_color_elements(self.df, general_color_key, column_name, False, 2))

            for shade_key in self.general_shades:
                aggregator_json['number_of_elements_of_general_shade_' + shade_key + '_from_' + column_name] = float(self.get_number_or_percentage_of_color_elements(self.df, shade_key, column_name, True, 3))
                aggregator_json['percentage_of_elements_of_general_shade_' + shade_key + '_from_' + column_name] = float(self.get_number_or_percentage_of_color_elements(self.df, shade_key, column_name, False, 3))

        for key1 in ['color_closest_str', 'color_closest_str_with_HLS']:
            for key2 in ['text_total_rate', 'text_length']:
                for operator_key in ['count', 'sum', 'mean', 'min', 'max', 'std']:
                    self.all_groupby_by_2_key(self.df, key1, key2, aggregator_json, operator_key)

        for tag_name in ['a', 'div', 'span']:
            for column_name in ['color_closest_str', 'background_color_closest_str', 'color_closest_str_with_HLS', 'background_color_closest_str_with_HLS']:
                self.groupby_tag_name(self.df, column_name, tag_name, aggregator_json)

    def all_groupby_by_2_key(self, df, key1, key2, aggregator_json, operator_key):
        if operator_key == 'count':
            new_df = df.groupby(key1)[key2].count()
        elif operator_key == 'sum':
            new_df = df.groupby(key1)[key2].sum()
        elif operator_key == 'mean':
            new_df = df.groupby(key1)[key2].mean()
        elif operator_key == 'min':
            new_df = df.groupby(key1)[key2].min()
        elif operator_key == 'max':
            new_df = df.groupby(key1)[key2].max()
        elif operator_key == 'std':
            new_df = df.groupby(key1)[key2].std()

        for col_dict in self.color_dict:
            for index, row in zip(new_df.index, new_df):
                if index == col_dict:
                    if str(row) != 'nan':
                        aggregator_json['group_' + key2 + "_by_filter_" + key1 + "_" + index + "_" + operator_key] = float(row)
                    else:
                        aggregator_json['group_' + key2 + "_by_filter_" + key1 + "_" + index + "_" + operator_key] = float(0.0)
                    break
            else:
                aggregator_json['group_' + key2 + "_by_filter_" + key1 + "_" + col_dict + "_" + operator_key] = float(0.0)

    def groupby_tag_name(self, df, key, tag_name, aggregator_json):#

        df_tag = df[df['Tag Name'] == tag_name]
        new_df = (df_tag.groupby(key)['Tag Name'].count())
        total_values_in_the_df = float(sum([new_df[color_index] for color_index in new_df.index]))
        for color_from_dict in self.color_dict:
            for color_index in new_df.index:
                if color_from_dict == color_index and total_values_in_the_df > 0:
                    aggregator_json['number_of_'+color_from_dict+'_elements_form_column_'+key+'_filtered_by_tag_' + tag_name] = float(new_df[color_index])
                    aggregator_json['percentage_of_' + color_from_dict + '_elements_form_column_' + key + '_filtered_by_tag_' + tag_name] = float((new_df[color_index])/total_values_in_the_df)
                    break
            else:
                aggregator_json['number_of_'+color_from_dict+'_elements_form_key_' + key + '_filtered_by_tag_'+tag_name] = float(0.0)
                aggregator_json['percentage_of_' + color_from_dict + '_elements_form_column_' + key + '_filtered_by_tag_' + tag_name] = float(0.0)

    def get_number_of_elements_from_file(self, df):
        return float(len(df))

    def get_number_of_different_tag_names_from_file(self, df):
        return float(len(set(df['Tag Name'])))

    def get_number_of_elements_from_file_by_parameter(self, df, p_str, column_name='Tag Name'):
        return float(len([el for el in df[column_name] if el == p_str]))

    def get_number_of_not_NaN_elements_from_file_by_parameter(self, df, column_name):
        return float(len(df[np.isfinite(df[column_name])][column_name]))

    def get_totals_from_file_by_parameter(self, df, column_name):
        return float(sum([float(tt) for tt in df[np.isfinite(df[column_name])][column_name]]))

    def get_number_or_percentage_of_color_elements(self, df, color_and_shade_key, column_name, is_number, operator_key):
        denominator = self.get_number_of_all_color_elements(df, column_name)
        if float(denominator) > 0:
            if operator_key == 1:
                if is_number:
                    return float(len([el for el in df[column_name] if el == color_and_shade_key]))
                else:
                    numerator = (len([el for el in df[column_name] if el == color_and_shade_key]))
                    return float(numerator/denominator)
            else:
                if is_number:
                    return float(len([el for el in df[column_name] if str(color_and_shade_key) in str(el)]))
                else:
                    numerator = (len([el for el in df[column_name] if str(color_and_shade_key) in str(el)]))
                    return float(numerator / denominator)
        else:
            return 0.0

    def get_general_colors_and_shades(self, shade):
        return set([color_key.split('_')[shade] for color_key, color_value in self.color_dict.items()])

    def get_number_of_all_color_elements(self, df, column_name):
        return float(len([el for el in df[column_name] if len(str(el)) > 3]))

    def get_entropy_by_column_name_and_key(self, df, column_name, key):
        list_of_p = []
        if key == 1:
            for specific_color_key, color_value in self.color_dict.items():
                list_of_p.append(self.get_number_or_percentage_of_color_elements(df, specific_color_key, column_name, False, 1))

        elif key == 2:
            for general_color_key in self.general_colors:
                list_of_p.append(self.get_number_or_percentage_of_color_elements(df, general_color_key, column_name, False, 2))

        elif key == 3:
            for shade_key in self.general_shades:
                list_of_p.append(self.get_number_or_percentage_of_color_elements(df, shade_key, column_name, False, 3))


        return float(-sum([np.log(p)*p for p in list_of_p if p>0]))

class ImageAggregator(object):
    def __init__(self, file_name, color_dict):
        self.file_name = file_name
        self.color_dict = color_dict
        self.im = Image.open(self.file_name)
        self.image_size = float(float(self.im.size[1])*float(self.im.size[0]))
        self.all_image_colors = colorgram.extract(file_name, 1000)
        self.all_color_proportion = {}
        self.get_all_colors_information()
        # print(self.image_size)

    def get_all_colors_information(self):
        color_proportion_list = []
        # extract the colors from the web to the closest color in the color dict
        for color in self.all_image_colors:
            element = (color.rgb[0], color.rgb[1], color.rgb[2])
            color_str = Parser.find_color(element, self.color_dict, False)
            color_proportion_list.append((color_str, color.proportion))

        #for every color in the color dict with the same key and it's proportion value to a list per key
        color_proportion_set = dict()

        for line in color_proportion_list:
            if line[0] in color_proportion_set:
                # append the new number to the existing array at this slot
                color_proportion_set[line[0]].append(line[1])
            else:
                # create a new array in this slot
                color_proportion_set[line[0]] = [line[1]]

        # sum all values in the list to one proportion value per color
        for key, value in color_proportion_set.items():
            color_proportion_set[key] = sum(value)

        # set values for all keys in original dict
        for color_dict_key in self.color_dict.keys():
            for key, value in color_proportion_set.items():
                if color_dict_key == key:
                    self.all_color_proportion["proportion_of_color_"+color_dict_key+"_in_the_image"] = float(value)
                    break
            else:
                self.all_color_proportion["proportion_of_color_"+color_dict_key+"_in_the_image"] = float(0.0)

        return True

    def get_all_color_proportion_in_image(self):
        return self.all_color_proportion

class CombaindImageFileAggregator(object):
    def __init__(self, json_of_file, json_of_image):
        pass

