import re
import os
import boto3
import random
import base64
import botocore
import numpy as np
import pandas as pd
import ipywidgets as widgets

from tqdm import tqdm
from io import BytesIO
from pprint import pprint
from ast import literal_eval
from PIL import ImageDraw, ImageFont, Image, ImageStat
from IPython.display import clear_output, HTML

import warnings
warnings.filterwarnings("ignore")

s3 = boto3.resource('s3')
s3_session = boto3.session.Session().client('s3')

class LabelOne:
    def __init__(
        self,
        input_csv=None,
        output_csv='output.csv',
        text_column=None,
        image_column=None,
        words_column=None,
        bboxes_column=None,
        labels_column=None,
        annotation_column='annotation',
        width='1100px',
        height='500px',
        edit_width='1500px',
        edit_height='200px',
        display_labelword_df=False
    ):

        if not input_csv:
            print('[INFO] Please pass `input_csv` e.g. train.csv')
            return
        
        self.out = widgets.Output()

        self.INPUT_CSV = input_csv
        self.OUTPUT_CSV = output_csv
        self.TEXT_COLUMN = text_column
        self.IMAGE_COLUMN = image_column
        self.WORDS_COLUMN = words_column
        self.BBOXES_COLUMN = bboxes_column
        self.LABELS_COLUMN = labels_column
        self.ANNOTATION_COLUMN = annotation_column
        self.CSV_INDEX = 0
        self.df = pd.read_csv(self.INPUT_CSV, skipinitialspace=False, encoding='utf-8')
        self.width, self.height, self.edit_width, self.edit_height = width, height, edit_width, edit_height
        self.DISPLAY_LABELWORD_DF = display_labelword_df

        self.text_area = widgets.Textarea(
            layout={'width': '100%', 'height': '100%'}, value='')
        self.text_box = widgets.VBox([self.text_area], layout={
                                     'width': width, 'height': height})

        self.image = self.image_to_widget(Image.new("RGB", (1, 1)))
        self.image_box = widgets.VBox([self.image], layout={
                                      'width': self.width, 'height': self.height, 'border': 'solid gray'})
        self.font = ImageFont.load_default()
        self.color_dict = {}

        self.edit_area = widgets.Textarea(
            layout={'width': '100%', 'height': '100%'}, value='')
        self.edit_box = widgets.VBox([self.edit_area], layout={
                                     'width': self.edit_width, 'height': self.edit_height})

        self.prev_button = widgets.Button(
            description='Previous', button_style='warning')
        self.format_button = widgets.Button(
            description='Format', button_style='info')
        self.save_button = widgets.Button(
            description='Save', button_style='success')
        self.index_field = widgets.Text(description="Index", value='0')
        self.max_count_label = widgets.Label(value=f"/ {self.df.shape[0]}")

        self.prev_button.on_click(self.on_previous_clicked)
        self.format_button.on_click(self.on_format_clicked)
        self.save_button.on_click(self.on_save_clicked)
        self.index_field.observe(self.index_change, names='value')

        #print('[input_csv]', self.INPUT_CSV)
        #print('[output_csv]', self.OUTPUT_CSV)
        #print('[text_column]', self.TEXT_COLUMN)
        #print('[annotation_column]', self.annotation_COLUMN)


    def on_format_clicked(self, b):
        self.edit_area.value = self.remove_leading_space(self.edit_area.value)
        self.edit_area.value = self.remove_empty_lines(self.edit_area.value)
        self.edit_area.value = self.remove_extra_whitespace(
            self.edit_area.value)

    def add_column(self, df, col_name, col_content=[]):
        if col_name not in df.columns:
            df[col_name] = [None] * len(df)
        return df

    def remove_extra_whitespace(self, text):
        lines = text.split('\n')
        lines = [re.sub(r'\s+', ' ', line) for line in lines]
        return '\n'.join(lines)

    def remove_leading_space(self, text):
        lines = text.split('\n')
        lines = [line.lstrip() for line in lines]
        return '\n'.join(lines)

    def remove_empty_lines(self, text):
        return '\n'.join([line for line in text.split('\n') if line.strip() != '']).strip()

    # Split s3 uri
    def split_s3_uri(self, s3_uri):
        bucket = s3_uri.split('/')[2]
        prefix = '/'.join(s3_uri.split('/')[3:-1])
        filename = s3_uri.split('/')[-1]
        return bucket, prefix, filename

    # Creates a grouping of words, boxes, labels in a list
    def get_word_bbox_label(self):
        return list(zip(
            literal_eval(self.df.iloc[[self.CSV_INDEX]]
                         [self.WORDS_COLUMN].iloc[0]),
            literal_eval(self.df.iloc[[self.CSV_INDEX]]
                         [self.BBOXES_COLUMN].iloc[0]),
            literal_eval(self.df.iloc[[self.CSV_INDEX]]
                         [self.LABELS_COLUMN].iloc[0])
        ))
    
    # Check if S3 file exists - used before we download from S3
    def check_s3_file_exists(self, bucket, key):
        try:
            s3_session.head_object(Bucket=bucket, Key=key)
        except botocore.exceptions.ClientError as e:
            return int(e.response['Error']['Code']) != 404
        return True
    
    # Check if local file exists
    def check_local_file(self, file_name):
        if os.path.isfile(file_name):
            return True
        else:
            return False

    # Gets the image from S3 and draws the boxes
    def get_image(self):
        path_img = None if self.df.iloc[[self.CSV_INDEX]][self.IMAGE_COLUMN].iloc[0] in [
            '', None, np.nan, np.NaN] else self.df.iloc[[self.CSV_INDEX]][self.IMAGE_COLUMN].iloc[0]
        if not path_img:
            #print('No image')
            return None

        #!aws s3 cp {path_img} tmp.jpg --quiet
        bucket, prefix = None, None
        if 's3://' in path_img:
            bucket, prefix, filename = self.split_s3_uri(path_img)
        
        if bucket and prefix and self.check_s3_file_exists(bucket, f'{prefix}/{filename}'):
            s3.Bucket(bucket).download_file(f'{prefix}/{filename}', 'tmp.jpg')
            image = Image.open('tmp.jpg')
        elif self.check_local_file(path_img):
            image = Image.open(path_img)
        else:
            image = Image.new("RGB", (1, 1))

        #print('Size', image.size)
        draw = ImageDraw.Draw(image)

        for word, bbox, label in self.get_word_bbox_label():
            word = str(word)
            label = str(label)
            if not bbox:
                continue
            if label not in self.color_dict:
                self.color_dict[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # if not self.is_grayscale(path_img) else (0)

            draw.rectangle(bbox, outline=self.color_dict[label])
            draw.text((bbox[0] + 10, bbox[1] - 10), label,
                      fill=self.color_dict[label], font=self.font)

        return image
    
    def is_grayscale(self, path="image.jpg"):
        im = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(im)
        
        if sum(stat.sum)/3 == stat.sum[0]: #check the avg with any element value
            return True #if grayscale
        else:
            return False #else its colour

    # Display an image object inside a widget layout
    def image_to_widget(self, image):
        image_type = 'PNG' if 'png' in str(type(image)).lower() else 'JPEG'
        buffered = BytesIO()
        image.save(buffered, format=image_type)
        img_widget = widgets.Image(
            value=buffered.getvalue(),
            format=image_type,
            width=image.width,
            height=image.height,
        )
        return img_widget

    def create_df_html(self):
        words, labels = [], []
        for word, bbox, label in self.get_word_bbox_label():
            words.append(word)
            labels.append(label)
        pd.set_option('display.max_colwidth', 500)
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        return pd.DataFrame(list(zip(labels, words)),
                columns=['Labels', 'Words']).to_html()


    def on_previous_clicked(self, b):

        self.CSV_INDEX -= 1
        if self.CSV_INDEX < 0:
            self.CSV_INDEX = len(self.df) - 1

        self.update_values()       
        self.index_field.value = str(self.CSV_INDEX + 1)
        
    def on_save_clicked(self, b):

        # Save stuff here
        self.df[self.ANNOTATION_COLUMN].fillna('', inplace=True)
        self.df.at[self.CSV_INDEX, self.ANNOTATION_COLUMN] = self.edit_area.value if self.edit_area.value != '' else ''
        self.df.to_csv(self.OUTPUT_CSV, index=False)

        # After saving, move on to the next item
        if self.CSV_INDEX >= len(self.df) - 1:
            self.CSV_INDEX = 0
        else:
            self.CSV_INDEX += 1

        self.update_values()     
        self.index_field.value = str(self.CSV_INDEX + 1)
    
    # Default send to out widget, and manually clear output in code
    #@out.capture()
    def index_change(self, change):
        #print('CHANGE', change)
        if change['new'] == '':
            return

        try:
            self.CSV_INDEX = int(change['new']) - 1
            self.update_values()
        except Exception as ex:
            pass
            #print(ex)
        
        with self.out:
            clear_output()
            display(self.main_window())

    # Updates values for the widgets
    def update_values(self, begin=False, increment=False):
        
        if self.ANNOTATION_COLUMN not in self.df.columns:
            self.df = self.add_column(self.df, self.ANNOTATION_COLUMN)

        if self.TEXT_COLUMN != None:
            self.text_area.value = '' if self.df.iloc[[self.CSV_INDEX]][self.TEXT_COLUMN].iloc[0] in [
                '', None, np.nan, np.NaN] else self.df.iloc[[self.CSV_INDEX]][self.TEXT_COLUMN].iloc[0]

        if self.IMAGE_COLUMN != None:
            self.image = self.image_to_widget(self.get_image())
            self.image_box = widgets.VBox([self.image], layout={
                                          'width': self.width, 'height': self.height, 'border': 'solid gray'})

        self.df[self.ANNOTATION_COLUMN].fillna('', inplace=True)
        self.edit_area.value = '' if self.df.iloc[[self.CSV_INDEX]][self.ANNOTATION_COLUMN].iloc[0] in [
            '', None, np.nan, np.NaN, 'nan'] else self.df.iloc[[self.CSV_INDEX]][self.ANNOTATION_COLUMN].iloc[0]

 
        #inc = 0 if not increment else 1
        if begin:
            self.index_field.value = str(self.CSV_INDEX + 1)

    # This decorator sets an output widget + clears the cell output before display
    #@out.capture(clear_output=True)
    def main_window(self, begin=False):

        content_box = []
        if self.TEXT_COLUMN != None:
            content_box.append(self.text_box)
        if self.IMAGE_COLUMN != None:
            content_box.append(self.image_box)
        if self.DISPLAY_LABELWORD_DF:
            content_box.append(widgets.HTML(value=self.create_df_html()))

        with self.out:
            clear_output()
            return widgets.VBox(
                [
                    widgets.HBox([
                        self.prev_button,
                        self.format_button,
                        self.save_button,
                        self.index_field,
                        self.max_count_label
                    ]),
                    self.edit_box,
                    widgets.HBox(content_box)
                ]
            )
    
    def load(self):
        display(self.main_window(begin=True))
        clear_output()
        self.update_values(begin=True)
        display(self.out)
