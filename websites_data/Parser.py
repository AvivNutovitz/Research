import re
import nltk
from bs4 import BeautifulSoup as soup
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
import contextlib
import premailer
import webcolors
import time
from PIL import Image
import math
import colorsys
import os

class WebParser(object):

    def __init__(self, url, color_dict):
        self.PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
        self.tags_black_list = ['script', 'noscript', 'style', '<script>', '<link href', '<script ', 'meta content']
        self.all_keys = ["position", "font-weight", "background-color", "color", "font-size", "width", "height"]
        self.full_image_path = './image_of_'+str(url).split('//')[1]+'.png'
        self.color_dict = color_dict
        self.df = pd.DataFrame()
        self.bold_factor = 1.5
        self.url_list = []
        self.elements_tag_names_list = []
        self.elements_attribute_class_list = []
        self.elements_attribute_id_list = []
        self.elements_attribute_style_list = []
        self.elements_text_data_list = []
        self.elements_get_computed_style_list = []
        self.main_process(url)

    def get_driver(self, is_chrome):
        chrome_path = self.PROJECT_ROOT+"\drivers\chromedriver.exe"
        firefox_path = self.PROJECT_ROOT+"\drivers\geckodriver.exe"
        if is_chrome:
            driver = webdriver.Chrome(executable_path=chrome_path)
        else:
            driver = webdriver.Firefox(executable_path=firefox_path)
        return driver

    def get_csv_file_name(self, url):
        if url[-1] == '/':
            return "file_from_" + url[:-1].split('//')[1] + ".csv"
        else:
            return "file_from_" + url.split('//')[1] + ".csv"

    def rgb_converter(self, df, col_name):
        list_to_convert = df[col_name]
        list_to_add = []
        list_original_color_data_type = []
        for color_inline in list_to_convert:
            try:
                #color is writen in HEX (1)
                if color_inline[0] == "#":
                    list_to_add.append(webcolors.hex_to_rgb(color_inline))
                    list_original_color_data_type.append('1')
                # color is writen in RGBA (2)
                elif "rgba(" in color_inline.lower():
                    list_to_add.append(color_inline)
                    list_original_color_data_type.append('2')
                elif " rgba(" in color_inline.lower():
                    list_to_add.append(color_inline)
                    list_original_color_data_type.append('2')
                # color is writen in RGB (3)
                elif "rgb(" in color_inline.lower():
                    list_to_add.append(color_inline)
                    list_original_color_data_type.append('3')
                elif " rgb(" in color_inline.lower():
                    list_to_add.append(color_inline)
                    list_original_color_data_type.append('3')
                # color is writen in words (4)
                else:
                    list_to_add.append(webcolors.name_to_rgb(color_inline))
                    list_original_color_data_type.append('4')
            except:
                list_to_add.append('NaN')
                list_original_color_data_type.append('NaN')

        df['RGB from '+ col_name] = list_to_add
        df['Original color data type of '+col_name] = list_original_color_data_type
        # df.drop([col_name], axis=1, inplace=True)
        return df

    def extract_text_from_element(self, element_text):
        try:
            #return the text of an element NOt his children
            if element_text.name in['body', 'style']:
                return 'NaN'
            else:
                words = element_text.find(text=True)
                # clean the string
                words = words.strip(' \t\n\r')
                if (len(words) > 0):
                    return words
                else:
                    return 'NaN'
        except:
            return None
        # text = element_text.text or ''
        # tail = element_text.tail or ''
        # words = ' '.join((text, tail)).strip()

    def update_all_keys(self, list_of_all_keys):
        for row in list_of_all_keys:
            all_style_properties_in_the_row = row.split(';')
            # contains key, value pairs separated by :
            for pair in all_style_properties_in_the_row:
                key = pair.split(':')[0]
                self.all_keys.append(key)

        return set(self.all_keys)

    def find_property_from_line(self, line, property_key):
        all_style_properties = line.split(';')
        # contains key, value pairs separated by :
        for pair in all_style_properties:
            key_form_row = pair.split(':')[0].replace(" ","")
            if (key_form_row == property_key):
                value = pair.split(':')[1]
                return value.strip()
            else:
                pass
        else:
            return 'NaN'

    def get_content_form_url(self, driver, url):
        content = None
        try:
            with contextlib.closing(driver) as browser:
                browser.get(url)  # Load page
                self.waitForLoad(browser)
                content = browser.page_source
                try:
                    browser.close()
                    browser.quit()
                except:
                    pass
        except:
            try:
                driver.close()
            except:
                pass
            try:
                driver.quit()
            except:
                pass

        # from command lis if needed
        # "taskkill /im chromedriver.exe /f"
        try:
            os.cmd("taskkill /im chromedriver.exe /f")
        except:
            pass


        return content

    def extract_all_elements_from_page(self, driver, url):
        html_page = soup(self.get_content_form_url(driver, url), "html.parser")
        ext_styles_before_retrieving = html_page.findAll('link', rel="stylesheet")
        ext_styles_links = [e_st['href'] for e_st in ext_styles_before_retrieving]
        [s.decompose() for s in html_page(self.tags_black_list)]
        str_tree = html_page.body.prettify()
        p = premailer.Premailer(str_tree, base_url=url, external_styles=ext_styles_links)

        result = p.transform()
        new_html_page = soup(result, "html.parser")
        elements = new_html_page.find_all()
        return elements

    def clear_all_lists(self):
        del self.url_list[:]
        del self.elements_tag_names_list[:]
        del self.elements_attribute_class_list[:]
        del self.elements_attribute_id_list[:]
        del self.elements_attribute_style_list[:]
        del self.elements_text_data_list[:]

    def main_loop(self, elements, url):
        for element in elements:
            try:
                try:
                    e_name = element.name
                except:
                    e_name = 'NaN'
                try:
                    e_id = element["id"]
                except:
                    e_id = 'NaN'
                try:
                    e_class = element['class']
                except:
                    e_class = 'NaN'
                try:
                    e_style = element['style']
                except:
                    e_style = 'NaN'

                self.url_list.append(url)
                self.elements_tag_names_list.append(e_name)
                self.elements_attribute_class_list.append(e_class)
                self.elements_attribute_id_list.append(e_id)
                self.elements_attribute_style_list.append(e_style)
                self.elements_text_data_list.append(self.extract_text_from_element(element))

            except:
                pass

    def get_text_attributes(self, df, elements_text_data_list):
        list_of_number_of_words = []
        list_of_text_length = []
        list_of_text_total_rate = []
        for index, text_data in enumerate(elements_text_data_list):
            if (text_data != 'NaN' and text_data is not None):
                num_of_charts = 0.0
                #Text length
                try:
                    if num_of_charts != None:
                        num_of_charts = float(len(text_data.replace(" ", "").strip()))
                except:
                    pass
                list_of_text_length.append(float(str(num_of_charts).strip()))
                #Number of words in the text
                try:
                    word_tokenize_list = nltk.word_tokenize(text_data)
                    list_of_number_of_words.append(float(len(word_tokenize_list)))
                except:
                    list_of_number_of_words.append(float(0.0))

                #Text total rate
                try:
                    if(df['font-weight'].get(index) != 'NaN' and df['font-weight'].get(index) is not None): # mean we have a bold element
                        list_of_text_total_rate.append(float(self.bold_factor*float(num_of_charts)))
                    else:
                        list_of_text_total_rate.append(float(num_of_charts))
                except:
                    pass
            else:
                list_of_text_length.append('NaN')
                list_of_number_of_words.append('NaN')
                list_of_text_total_rate.append('NaN')

        df['text_length'] = list_of_text_length
        df['number_of_words'] = list_of_number_of_words
        df['text_total_rate'] = list_of_text_total_rate
        return df

    def get_element_area(self, df):

        element_height_list = df['height']
        element_width_list = df['width']
        area_list = []
        for height, width in zip(element_height_list, element_width_list):
            if (height != 'NaN')and(width != 'NaN')and(('%' or 'auto') not in height)and(('%' or 'auto') not in width):
                try:
                    area_list.append(float(re.findall(r'\d+', height)[0]) * float(re.findall(r'\d+', width)[0]))
                except:
                    area_list.append('NaN')
            else:
                area_list.append('NaN')
        df['element_area'] = area_list
        return df

    def update_df(self, df, all_keys, url):
        df['Url'] = self.url_list
        df['Tag Name'] = self.elements_tag_names_list
        df['Class'] = self.elements_attribute_class_list
        df['Id'] = self.elements_attribute_id_list
        df['Style'] = self.elements_attribute_style_list
        df['Text'] = self.elements_text_data_list

        for key in all_keys:
            df[key] = [self.find_property_from_line(line, key) for line in df['Style']]

        #convert background-color to RGB
        df = self.rgb_converter(df, "background-color")

        #convert color to RGB
        df = self.rgb_converter(df, "color")

        #text manipulation
        df = self.get_text_attributes(df, self.elements_text_data_list)

        #calc element area
        df = self.get_element_area(df)

        #closest color to rgb data
        df = self.get_closest_color(df, self.color_dict)

        df.drop(['Style'], axis=1, inplace=True)
        df.to_csv(readable_url(url, False))
        return df

    def fullpage_screenshot(self, driver, file):

        print("Starting chrome full page screenshot workaround ...")

        print(str(driver.desired_capabilities))
        # print(str(driver.get_window_size()))
        total_width, total_height, viewport_width, viewport_height = self.get_driver_window_params(driver)
        print(
            "Total: ({0}, {1}), Viewport: ({2},{3})".format(total_width, total_height, viewport_width, viewport_height))
        rectangles = []

        i = 0
        while i < total_height:
            ii = 0
            top_height = i + viewport_height

            if top_height > total_height:
                top_height = total_height

            while ii < total_width:
                top_width = ii + viewport_width

                if top_width > total_width:
                    top_width = total_width

                print("Appending rectangle ({0},{1},{2},{3})".format(ii, i, top_width, top_height))
                rectangles.append((ii, i, top_width, top_height))

                ii = ii + viewport_width

            i = i + viewport_height

        stitched_image = Image.new('RGB', (total_width, total_height))
        previous = None
        part = 0
        last_offset = 0

        for rectangle in rectangles:

            if not previous is None:
                driver.execute_script("window.scrollTo({0}, {1})".format(rectangle[0], rectangle[1]))
                print("Scrolled To ({0},{1})".format(rectangle[0], rectangle[1]))
                time.sleep(0.3)

            file_name = "part_{0}.png".format(part)
            print("Capturing {0} ...".format(file_name))

            driver.get_screenshot_as_file(file_name)
            screenshot = Image.open(file_name)

            if rectangle[1] + viewport_height > total_height:
                offset = (rectangle[0], total_height - viewport_height)
            else:
                width_temp, height_temp = screenshot.size
                offset = (rectangle[0], last_offset)
                last_offset += height_temp

            print("Adding to stitched image with offset ({0}, {1})".format(offset[0], offset[1]))
            stitched_image.paste(screenshot, offset)

            del screenshot
            # os.remove(file_name)
            part = part + 1
            previous = rectangle

        stitched_image.save(file)
        print("Finishing chrome full page screenshot workaround...")
        return True

    def get_right_rectangles_to_screen_shot(self, rectangles):
        correct_list  = []
        upper_left_corner = 0
        upper_right_corner = 0
        lower_left_corner = 0
        lower_right_corner = 0
        for rec in rectangles:
            # find if 3 corners are the same, if not add to list
            if (upper_left_corner == rec[0] and upper_right_corner == rec[1] and lower_left_corner == rec[2])\
                    or(upper_right_corner == rec[1] and lower_left_corner == rec[2] and lower_right_corner == rec[3])\
                    or(lower_left_corner == rec[2] and lower_right_corner == rec[3] and upper_left_corner == rec [0])\
                    or(lower_right_corner == rec[3] and upper_left_corner == rec[0] and upper_right_corner == rec[1]):
                pass
            else:
                correct_list.append(rec)

            upper_left_corner = rec[0]
            upper_right_corner = rec[1]
            lower_left_corner = rec[2]
            lower_right_corner = rec[3]
        return correct_list

    def get_driver_window_params(self, driver):

        # total_height = driver.execute_script("return document.body.parentNode.scrollHeight")
        # viewport_height = driver.execute_script("return window.innerHeight")
        #
        # temp_scroll = 0
        # while (True):
        #     if temp_scroll < total_height:
        #         driver.execute_script("window.scrollTo(0,"+str(temp_scroll)+")")
        #         time.sleep(0.3)
        #         temp_scroll += viewport_height
        #     else:
        #         driver.execute_script("window.scrollTo(0,0)")
        #         break

        total_width = driver.execute_script("return document.body.offsetWidth")
        total_height = driver.execute_script("return document.body.parentNode.scrollHeight")
        viewport_width = driver.execute_script("return document.body.clientWidth")
        viewport_height = driver.execute_script("return window.innerHeight")

        # driver.get_screenshot_as_file('temp.png')
        # width_temp, height_temp = Image.open('temp.png').size
        # os.remove('temp.png')
        #
        # if (width_temp > viewport_width):
        #     viewport_width = width_temp
        #
        # if (width_temp > total_width):
        #     total_width = width_temp
        #
        # if (height_temp > viewport_height):
        #     viewport_height = height_temp
        #
        # if (height_temp > total_height):
        #     total_height = height_temp

        return total_width, total_height, viewport_width, viewport_height

    def get_closest_color(self, df, color_dict):
        RGB_from_background_color_list = df['RGB from background-color']
        RGB_from_color_list = df['RGB from color']

        df['background_color_closest_str'] = [find_color(bgc, color_dict, False) for bgc in RGB_from_background_color_list]
        df['color_closest_str'] = [find_color(bgc, color_dict, False) for bgc in RGB_from_color_list]

        df['background_color_closest_str_with_HLS'] = [find_color(bgc, color_dict, True) for bgc in RGB_from_background_color_list]
        df['color_closest_str_with_HLS'] = [find_color(bgc, color_dict, True) for bgc in RGB_from_color_list]

        df['background_color_HLS_L'] = [self.get_HLS_values(bgc, 'L') for bgc in RGB_from_background_color_list]
        df['color_HLS_L'] = [self.get_HLS_values(col, 'L') for col in RGB_from_color_list]

        df['background_color_HLS_S'] = [self.get_HLS_values(bgc, 'S') for bgc in RGB_from_background_color_list]
        df['color_HLS_S'] = [self.get_HLS_values(col, 'S') for col in RGB_from_color_list]

        df['background_color_HLS_H'] = [self.get_HLS_values(bgc, 'H') for bgc in RGB_from_background_color_list]
        df['color_HLS_H'] = [self.get_HLS_values(col, 'H') for col in RGB_from_color_list]

        return df

    def main_process(self, url):

        driver = self.get_driver(True)
        elements = self.extract_all_elements_from_page(driver, url)
        # all_keys = update_all_keys(elements_attribute_style_list)

        self.main_loop(elements=elements, url=url)
        self.df = self.update_df(self.df, self.all_keys, url)
        #clear the data frame
        self.df.drop(self.df.index, inplace=True)
        #reorder columns
        # self.df.reindex(sorted(self.df.columns), axis=1)
        self.df.reindex(sorted(self.df.columns))
        #clear lists
        self.clear_all_lists()

        # browser = self.get_driver(False)
        # browser.maximize_window()
        # browser.get(url)
        # browser.save_screenshot(self.full_image_path)
        # browser.quit()

    def get_HLS_values(self, element, hsl):

        if str(element) == 'NaN':
            return 'NaN'
        else:
            if 'rgb' in str(element) or ' rgb' in str(element):
                element = tuple(map(int, str(element).split('(')[1].split(')')[0].split(',')))

            (r1, g1, b1) = element
            (h1, l1, s1) = colorsys.rgb_to_hls(r1, g1, b1)
            if str(hsl) == 'L':
                return l1
            elif str(hsl) == 'S':
                return s1
            else:
                return h1

    def waitForLoad(self, driver):
        # elem = driver.find_element_by_tag_name("html")
        elem = None
        count = 0
        while True:
            count += 1
            if count > 30:
                print("Timing out after 15 seconds and returning")
                return
            time.sleep(.5)
            try:
                elem == driver.find_element_by_tag_name("html")
            except StaleElementReferenceException:
                return

def find_color(element, color_dict, HSL_converter):
    color_list = []
    if str(element) == 'NaN':
        return 'NaN'
    else:
        for color_key, color_value in color_dict.items():
            color_list.append((color_key, calc_dist_from_element_to_base_color(element, color_value, HSL_converter)))
        temp_list = [color for color, distance in sorted(color_list, key=lambda x: x[1])]
        return temp_list[0]

def calc_dist_from_element_to_base_color( element, base_color, HSL_converter):
    #clean up
    if 'rgb' in str(element) or ' rgb' in str(element):
        element = tuple(map(int, str(element).split('(')[1].split(')')[0].split(',')))

    (r1, g1, b1) = element
    (r2, g2, b2) = base_color

    if not HSL_converter:
        return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
    else:
        (h1, l1, s1) = colorsys.rgb_to_hls(r1, g1, b1)
        (h2, l2, s2) = colorsys.rgb_to_hls(r2, g2, b2)
        # return math.sqrt((h1 - h2) ** 2 + (l1 - l2) ** 2 + (s1 - s2) ** 2)
        #determint the closest color only by h of hsl
        return math.sqrt((h1-h2)**2)

def readable_url(url, is_parser):
    # "file_from_" + url[:-1].split('//')[1] + ".csv"
    if (is_parser): # need the original url
        return url
    else: # need to write and read file_name format
        url_0 = str(url.split('//')[1]).replace(".", "_").replace("-", '_')
        if url_0[-1] == '/':
            return "file_from_" + url_0[:-1] + ".csv"
        else:
            return "file_from_" + url_0 + ".csv"