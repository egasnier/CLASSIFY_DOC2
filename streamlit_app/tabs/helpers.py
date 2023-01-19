from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2
from IPython.display import Markdown, display
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from transformers import pipeline
import tensorflow as tf


encoder = {
    0: 'Budget',
    1: 'Email',
    2: 'File folder',
    3: 'Id piece',
    4: 'Invoice',
    5: 'Other types',
    6: 'Passport',
    7: 'Pay',
    8: 'Postcard',
    9: 'Questionnaire',
    10: 'Residence proof',
    11: 'Resume',
    12: 'Scientific doc',
    13: 'Specification'
}


def get_cv2_image_from_upload(file):
    file_bytes = np.asarray(bytearray(file), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    return image

def read_image(filename):
  return cv2.imread('../data/final/' + filename)


fr_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

def normalize_before_translate(text):
    text = re.sub(r'([A-ZÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]+)',replacement, text)
    text = re.sub("\d+[\.,/]?", "", text)
    text = re.sub("[<>*%]?", "", text)
    return text

def translate_fr_to_en(text,verbose=True):
    try:
        text = normalize_before_translate(text)
        result = fr_en_translator(text,max_length=2000)[0]['translation_text']
        if verbose:
            print('From : ',text[0:30],' ... To:',result[0:30],' ...')
      
    except:
        print('Except try by phrase ')
        phrases = text.split('.')
        result = ''
        for phrase in phrases:
            try:
                tphrase = fr_en_translator(phrase,max_length=2000)[0]['translation_text']
                if verbose:
                    print('From phrase: ',phrase[0:30],' ... To:',tphrase[0:30],' ...')
                result = result + tphrase + '.' 
            except:
                print('Except on phrase ',phrase[0:30])  

    return result




def ocr_predict(filename,bin=False):
    for i in range(3):
        try:
            if (bin):image= DocumentFile.from_images(filename)
            else:
                print('Process',filename)
                image = DocumentFile.from_images('../data/final/' + filename)
            predictor = ocr_predictor(pretrained=True)
            result = predictor(image)
        except KeyError as e:
            if i < 2: 
                print('Retry',i)
                continue
            else:
                raise
        break
    return result

def ocr_get_text(document):
    obj = document.export()
    text = ''
    for page in obj['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    text = text + ' ' + word['value']
    return text

def image_to_string(filename,show=False):  
    result = ocr_predict(filename) 
    if (show):
        result.show(image)
    return ocr_get_text(result)

def show_sample_image_ocr(df):
    row = df.sample()
    filename = row['filename'].values[0]
    data_type = row['type'].values[0]
    print(filename,data_type)
    img = read_image(filename)
    print(image_to_string(filename,True))

def load_image(image, preprocess, channels = 3, size = (224, 224)):
    # Decoding
    img = tf.image.decode_jpeg(image, channels = channels)
    # Resizing
    img = tf.image.resize(img, size = size, method = 'nearest')
    # Casting
    img = tf.cast(img, tf.float32)
    if preprocess:
        img = preprocess(img)
    return img   

def printmd(string):
    display(Markdown(string))
    
def show_wrong_prediction(df): 
    fig = plt.figure(figsize=(20., 20.))
    
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                 axes_pad=0.8,  # pad between axes in inch.
                 aspect = True
                 )
    printmd('## Wrong predictions for ' + df.iloc[0,:]['type'])
    for ax,(index,row) in zip(grid,df.iterrows()):
        ax.imshow(read_image(row['filename']))
        ax.set_title('Detected as ' + row['pred'])
        ax.grid(False)
    plt.show()  

def show_wrong_predictions(df,y_test,y_pred):   
    
    X_result = df.iloc[y_test.index,:][['filename','text','type']].copy()
    X_result['real'] = y_test
    X_result['pred'] = y_pred
    X_bad = X_result[X_result['pred']!=X_result['real']]
    
    fig = plt.figure(figsize=(10,10))
    sns.countplot(y='type',data=X_bad)
    plt.title('Wrong predictions count')
    plt.show()
    
    
    vc = pd.DataFrame(X_bad['type'].value_counts()).head(8)
    for t in vc.index.tolist():
        show_wrong_prediction(X_bad[X_bad['type']==t])