import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import re
from .helpers import ocr_predict,read_image,ocr_get_text
from doctr.utils.visualization import visualize_page
import numpy as np
import cv2
from PIL import Image

title = "Extraction d'information"
sidebar_name = "Extraction d'information"
st.set_option('deprecation.showfileUploaderEncoding', False)

def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

def extractInfo(text):  
    idNr = re.findall(r'DocumentNo.\s(.*?)\s',text)
    idLastName = re.findall(r'NOM/S[^\s].*?\s+(.*?)\s',text)
    idFirstName = re.findall(r'Given[^\s].*?\s+(.*?)SEX',text)

    if (len(idNr)==0):
        idNr = re.findall(r'No:\s(.*?)\s',text)
        idLastName = re.findall(r'(?:Nom|NOM):\s+(.*?)\s',text)
        idFirstName = re.findall(r'Pr.*?:\s*(.*?)\s',text)
    
    if (len(idNr)==0):
        idNr = re.findall(r'N:\s(.*?)\s',text)
        idLastName = re.findall(r'(?:Nom|NOM):\s+(.*?)\s',text)
        idFirstName = re.findall(r'Pr.*?:\s*(.*?)\s',text)

    IDNR=''
    FIRSTNAME=''
    LASTNAME=''
    if (len(idNr)!=0): IDNR=idNr[0]
    if (len(idLastName)!=0): LASTNAME=idLastName[0]
    if (len(idFirstName)!=0): FIRSTNAME=idFirstName[0]

    st.markdown(f"""Bonjour **{FIRSTNAME} {LASTNAME}**.
         Votre piece d'identité n°**{IDNR}** a été transmis pour la verification""")

def processImage(filename,bin=False):
    with st.spinner("Analyzing..."):
            if (bin): 
                file_bytes = np.asarray(bytearray(filename), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
            else : 
                image = read_image(filename)

            document = ocr_predict(filename,bin)
            fig = visualize_page(document.pages[0].export(), image, interactive=False,add_labels=False)
            st.pyplot(fig)
            texte = ocr_get_text(document)
            #st.markdown(texte)
            extractInfo(texte)


def run():
    st.title(title)

    df = pd.read_csv('../data/data_with_meta.csv')
 
    st.markdown("""
     Pour l'extraction d'information à partir des pièces d'identités on utilise
     le texte extrait via le processus d'océrisation sur lequel on applique des expressions 
     régulières pour localiser les zones de textes qui nous intéressent. 
     En fonction de type de document on applique différentes expressions régulières
     """)

    good = ['img_0000076.jpg','img_0000081.jpg']

    df_id = df[df['filename'].isin(good)]
    row = df_id.sample()

    if st.button("Essaye avec un exemple"):
        doctype = row['type'].values[0]
        filename = row['filename'].values[0]
        st.markdown(f"""***Filename:{filename} Type:{doctype}***""")
        processImage(filename)

    st.markdown("**OU**")
    uploaded_file = st.file_uploader("Choisir une image")
    if uploaded_file is not None:
        # To read file as bytes:
        processImage(uploaded_file.read(),True)
   


