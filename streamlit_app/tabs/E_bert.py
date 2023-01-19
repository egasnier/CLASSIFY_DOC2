import streamlit as st
import pandas as pd
import numpy as np

from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizerFast, BertForSequenceClassification
from .helpers import ocr_predict,read_image,ocr_get_text,get_cv2_image_from_upload,translate_fr_to_en
from doctr.utils.visualization import visualize_page
import transformers
import sys


""""
import fasttext
fmodel = fasttext.load_model('../data/lid.176.bin')
def get_language_code_fasttext(text):
    try:
        return fmodel.predict(text.lower())[0][0].split("__label__")[1]
    except Exception as e:
        return 'unknown'
"""
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


def predictImage(image):
    with st.spinner("Ouverture..."):
        model = BertForSequenceClassification.from_pretrained('../model/bert_nlp_target_min')
        trainer = Trainer(
            model=model
        )
        cv2image = get_cv2_image_from_upload(image)
        st.markdown("### Traitement de l'image")
        st.image(cv2image)
    with st.spinner("Extraction OCR en cours..."):
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)
        document = ocr_predict(image,bin)
        text = ocr_get_text(document)
        fig = visualize_page(document.pages[0].export(), cv2image, interactive=False,add_labels=False)
        st.markdown("### Extraction OCR")
        st.pyplot(fig)
        st.markdown(f"Text : {text}")
    #st.markdown("### Detection langue")
    #code_lang = get_language_code_fasttext(text)
    #st.markdown(f"Langue détécté : {code_lang}")
    with st.spinner("Traduction en cours.."):
        text = translate_fr_to_en(text,False)
        st.markdown("### Traduction")
        st.markdown(f"Text traduit en anglais: {text}")
    with st.spinner("Prédiction en cours.."):
        max_length=512
        t = tokenizer(text, truncation=True, padding=True, max_length=max_length)
        y_pred = trainer.predict([t])
        predictions = y_pred.predictions
        predict = np.argmax(predictions, axis=-1)
        pred = encoder[predict[0]]
        prob = np.max(predictions, axis=-1) 
        st.markdown("### Prédiction")
        #st.markdown(f"Classifie en tant que **{pred}** (précision: {prob[0]})")
        st.markdown(f"Classifie en tant que **{pred}** ")
    return (document,text,pred)

title = "Bert (Bidirectional Encoder Representations from Transformers)"
sidebar_name = "Bert"


def run():
    st.title(title)

    df = pd.read_csv('../data/data_with_meta.csv')
    st.markdown("""
      Pour cette étape, nous utiliserons le modèle BERT.
      Ce modèle est conçu pour être pré-entraîné sur des représentations bi-directionelles 
      à partir d'un texte non labellisé en conditionnant sur le contexte de droite et de gauche
    (bi-directionnel) sur chaque couche du modèle.
    """)
    st.markdown("---")
    st.markdown("### Résultats obtenus")
    st.image(
        "./assets/bert_score.jpg"
    )
    st.markdown("""
        On remarque une très bonne précision sur les passeports (0.92) 
        et pièces d'identité (0.81) avec un rappel de 0.71 sur les passeports et 0.96
        sur les pièces d'identités.
    """)
   

    st.markdown("### Tester le model")
    uploaded_file = st.file_uploader("Choisir une image")
    if uploaded_file is not None:
        # To read file as bytes:
        image = uploaded_file.read()
        (document,text,predict) = predictImage(image)
            
            
        
        
        

