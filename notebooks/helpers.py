from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2
from IPython.display import Markdown, display
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn import metrics
import itertools
from sklearn.metrics import classification_report
import os.path
import numpy as np

def read_image(filename):
  return cv2.imread('../data/final/' + filename)

def ocr_predict(filename):
    for i in range(3):
        try:
            print('Process',filename)
            predictor = ocr_predictor(pretrained=True)
            image = DocumentFile.from_images('../data/final/' + filename)
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
        image = DocumentFile.from_images('../data/final/' + filename)
        result.show(image)
    return ocr_get_text(result)

def show_sample_image_ocr(df):
    row = df.sample()
    filename = row['filename'].values[0]
    data_type = row['type'].values[0]
    print(filename,data_type)
    img = read_image(filename)
    print(image_to_string(filename,True))
    

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
    
    X_result = df.iloc[y_test.index,:][['filename', 'type']].copy()
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

"""
METHOD FOR LOADING AN IMAGE
INPUT:
- directory  => Directory where is located the image
- filename   => Name of the file
- preprocess => Preprocessing method (tf.keras.applications.vgg16.preprocess_input for instance for VGG16 algorithm)
- channels   => Number of channels, 3 by default (color)
- size       => New image resolution, (224, 224) by default
OUTPUT:
- Image in a readable format for "plt.imshow" or for Computer Vision algorithms

EXAMPLE: Display a document randomly
IMAGES_DIRECTORY = '../data/final/'
from tensorflow.keras.applications.efficientnet import preprocess_input
num_alea = np.random.randint(len(df))

plt.imshow(load_image(directory = IMAGES_DIRECTORY,
                      filename = df.filename[num_alea],
                      preprocess = preprocess_input).numpy().astype("uint8"));
"""
def load_image(directory, filename, preprocess, channels = 3, size = (224, 224)):
    # Reading
    img = tf.io.read_file(directory + filename)
    # Decoding
    img = tf.image.decode_jpeg(img, channels = channels)
    # Resizing
    img = tf.image.resize(img, size = size, method = 'nearest')
    # Casting
    img = tf.cast(img, tf.float32)
    if preprocess:
        img = preprocess(img)
    return img



"""
METHOD TO GRAPHICAL DIPLAY THE CONFUSION MATRIX 
"""

def confusion_matrix(y_test, y_pred,classes=[]):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    if (len(classes)==0):
        extract_classes = pd.DataFrame(data = y_test.unique(), columns = ['classe']).sort_values(by = 'classe', ascending = True)
        classes = extract_classes.classe #range(0,21)

    plt.figure(figsize = (12, 12))

    plt.imshow(cnf_matrix, interpolation = 'nearest', cmap = 'Blues')
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 90)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment = "center",
                 color = "white" if cnf_matrix[i, j] > ( cnf_matrix.max() / 2) else "black")

    plt.ylabel('True labels')
    plt.xlabel('Labels predicted')
    plt.show()

"""
DISPLAY AND SAVE THE CLASSIFICATION REPORT 
"""
def global_classification_report(y_test, y_pred,classifier="None",category="None"):    
    append = False
    if(os.path.exists('../data/classification_report.csv')):
        df = pd.read_csv('../data/classification_report.csv')
        append = True
  
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose().reset_index() 
    df_report['classifier'] = classifier
    df_report['category'] = category    
    
    if (append):
        df = df.drop(df[(df['classifier']==classifier) & (df['category']==category)].index)
        df = pd.concat([df, df_report])
    else:
        df = df_report
        
    df.to_csv('../data/classification_report.csv',index=False)
    print(classification_report(y_test, y_pred))
    
"""
METHOD TO GROUP CLASSES 
"""
# Replace current classes by new classes i.e by grouping certain classes together
def replace_target(df):
    new_classes =  { 'advertisement': 'other_types',
                    'form'        : 'other_types',
                   'handwritten' : 'other_types',
                   'letter' : 'other_types',
                    'memo': 'other_types',
                    'presentation': 'other_types',
                   'invoice' :'facture',
                   'new_article': 'scientific_doc',
                   'scientific_publication': 'scientific_doc',
                   'scientific_report': 'scientific_doc'}

    df = df.replace(new_classes)
    # Creating dictionary
    dico_type = {'facture': 0,
                 'id_pieces': 1,
                 'justif_domicile': 2,
                 'passeport': 3,
                 'paye': 4,
                 'carte postale': 5,
                 'other_types': 6,
                 'scientific_doc': 7,
                 'resume': 8,
                 'specification': 9,
                 'budget': 10,
                 'file_folder': 11,
                 'email': 12,
                 'questionnaire': 13}

    # Creating revert dictionary
    dico_type_inv = {v: k for k, v in dico_type.items()}

    # Convert types according to the dictionary
    df['type_num'] = df.type.replace(dico_type)
    df.head()
