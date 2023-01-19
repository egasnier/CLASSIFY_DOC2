import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from .helpers import ocr_predict,read_image,ocr_get_text
from doctr.utils.visualization import visualize_page


title = "Preprocessing"
sidebar_name = "Preprocessing"


def run():
    st.title(title)

    df = pd.read_csv('../data/data.csv')
     
    st.markdown("###  1. Extraction du texte")
    st.markdown(""" 
    L'Optical Character Recognition (Reconnaissance Optique des Caractères ou Océrisation) 
    est un procédé permettant d’extraire du texte depuis une image dans un fichier texte.\n
    L'OCR est particulièrement utile pour dématérialiser des documents papiers imprimés et les remplacer
    par des bases de données exploitables par ordinateur.\n
     
    On utilise la bibliotèque [doctr](https://github.com/mindee/doctr) pour la reconnaissance optique.
     """)

    st.markdown("""Chaque image de notre data frame va être analysée et le texte correspondant va être sauvegardé
    dans une nouvelle colonne.""")
  
    row = df.sample()

    if st.button("Exemple"):
        doctype = row['type'].values[0]
        filename = row['filename'].values[0]
        st.markdown(f"""***Filename:{filename} Type:{doctype}***""")
        with st.spinner("Analyzing..."):
            image = read_image(filename)
            document = ocr_predict(filename)
            fig = visualize_page(document.pages[0].export(), image, interactive=False,add_labels=False)
            st.pyplot(fig)
            text = ocr_get_text(document)
            st.markdown(f"""####  Texte extrait
                        {text}""")
  
    st.markdown("---")
    st.markdown("### 2. Traduction")
    st.markdown("""
      Pour être capable de traiter de manière uniforme l'ensemble des données nous
      n'allons nous concentrer que sur les documents français et anglais.\n
      Une traduction française vers anglaise a été effectuée en utilisant la bibliothèque 
      transformer avec l'aide du modèle Helsinki-NLP/opus-mt-fr-en.\n
      Pour la traduction anglais francais on a utilisé le pipeline translation_en_to_fr
    """)

    if st.button(label="Exemple de traduction"):
      row = df.sample()
      st.markdown("###  " + row['type'].values[0])
      st.image('../data/final/' + row['filename'].values[0])

      if (row['lang_code'].values[0] == 'fr'):
        st.markdown("### Exemple de text en francais traduit en anglais")
        st.markdown("**Texte extrait en francais :** " +  str(row['text_ocr'].values[0])[0:160])
        st.markdown("**La traduction en anglais :** " +  str(row['text_en'].values[0])[0:160])                  
      else :
        st.markdown("### Exemple de text en anglais traduit en francais")
        st.markdown("**Texte extrait en anglais :** " +  str(row['text_ocr'].values[0])[0:160])
        st.markdown("**La traduction en francais :** " +  str(row['text_fr'].values[0])[0:160])      

      if st.button(label="Hide"):
          st.markdown("Hide")

    st.markdown("---")
    st.markdown("### 3. Suppression de stop words, ponctuations, tokenisation")
    if st.button(label="Voir la normalisation"):
       if (row['lang_code'].values[0] == 'fr'):
        st.markdown("**Texte extrait en francais :** " +  str(row['text_ocr'].values[0])[0:160])                 
       else :
        st.markdown("**Texte extrait en anglais :** " +  str(row['text_ocr'].values[0])[0:160])

       st.markdown("**Texte normalizé en anglais :** " +  str(row['text_en_norm'].values[0])[0:160])
       st.markdown("**Texte normalizé en francais :** " +  str(row['text_fr_norm'].values[0])[0:160]) 
    
    
    st.markdown("---")
    st.markdown("### 4. Regroupement des classes")
    st.markdown("""En analysant les labels, nous nous sommes aperçus de certaines similitudes.
     Ainsi, nous avons regroupé certaines classes.""")
    st.markdown("""
        | Ancienne classe  |  Nouvelle classe | 
        |---|---|
        |  advertisement | other_types  | 
        |  form | other_types  | 
        |  handwritten | other_types  | 
        |  letter | other_types  | 
        |  memo | other_types  | 
        |  presentation | other_types  | 
        |  news article | scientific_doc  | 
        |  scientific publication | scientific_doc  | 
        |  scientific report | scientific_doc  | 
    """)
    st.markdown("\n")
    st.markdown("""
     Par la suite pour l'approche NLP nous n'utilisons que les données en langue
     anglaise et française. Après le nettoyage notre dataset se réduit à 1177 lignes (1308 initialement).
    """)

