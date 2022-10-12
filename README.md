# labelone

Simple bounding box viewer for Jupyter. Click [here](https://chrispogeek.medium.com/bounding-box-viewer-widget-for-jupyter-913279265bca) for the blog post.

(Click the image below to watch a short demo)

[<img src="https://i.postimg.cc/Gh98FDN4/Label-One-Screenshot.png">](https://www.loom.com/share/50b14ccbddcc44ca9b44b798b39eb280)

### To use

- Get `requirements.txt` and run:

      pip install -r requirements.txt

- Get `labelone.py` and use in Jupyter as below:

      from labelone import LabelOne
      LabelOne(
            input_csv='input.csv',
            output_csv='output.csv',
            text_column='result',
            image_column='s3_uri',
            annotation_column='comment',
            words_column='tokens',
            bboxes_column='bboxes',
            labels_column='ner_tags',
            #width='300px',   #         <------ Adjust the UI size accordingly if needed
            #height='200px',
            #edit_width='900px',
            #edit_height='45px',
            display_labelword_df=True
      ).load()
      
  where:
  - `input_csv` - source csv file (local or S3 uri)
  - `output_csv` - destination csv file
  - `text_column` (Optional) - name of source csv/df column with text source
  - `image_column` (Optional) - input csv column name with rows containing path to the image
  - `annotation_column` (Optional) - name of new column for annotations
  - `words_column` - input csv column name with each row containing a list of words. See csv above for example
  - `bboxes_column` - input csv column name with each row containing a list of bounding boxes. See csv above for example
  - `labels_column` - input csv column name with each row containing a list of labels. See csv above for example
  - `width` (Optional) - width pixel size for text and image box content
  - `height` (Optional) - height pixel size for text and image box content
  - `edit_width/height` (Optional)- width/height for the annotation text box
  - `display_labelword_df` (Optional) - show a dataframe of labels and words (uses the words and labels columns)

  Sample input csv looks like this (included in this repo):
  
  ![](https://i.postimg.cc/NjPTH4bC/sampleinputcsv.png)
  
  ### Notes
  
  - Tested with Python 3.8 + JupyterLab 3.4.7
  
