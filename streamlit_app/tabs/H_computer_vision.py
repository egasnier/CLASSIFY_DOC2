import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from .helpers import load_image, get_cv2_image_from_upload, encoder
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input


INPUT_SHAPE      = (224, 224, 3)     # Resolution of images with channels before training


title = "Computer Vision"
sidebar_name = "Computer Vision"


def predictImage(image):
    
    with st.spinner("Chargement de l'image..."):
        cv2image = get_cv2_image_from_upload(image)
        st.markdown("### Traitement de l'image")
        st.image(cv2image)

    with st.spinner("Chargement du modèle..."):
        # Loading
        resnet = ResNet50(include_top = False, input_shape = INPUT_SHAPE)

        # Freezing weigths
        for layer in resnet.layers:
            layer.trainable = True
    
        # Building the model
        model = Sequential()
        model.add(resnet)
        model.add(GlobalAveragePooling2D()) 
        model.add(Dense(units = 1024, activation = 'relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units = 512, activation = 'relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units = 14, activation = 'softmax'))
    
        # Loading the weights
        model.load_weights('../model/ResNet50_target_min/ResNet50')

    with st.spinner("Prédiction en cours.."):
        X_test = load_image(image = image, preprocess = preprocess_input)
        y_prob = model.predict(np.array([X_test], dtype = np.float32))[0]

        # Class predictions
        predict = tf.argmax(y_prob, axis = -1).numpy()
        pred = encoder[predict]
        st.markdown("### Prédiction")
        st.markdown(f"Image classifiée en tant que **{pred}** ")
    return (pred)


def run():
    st.title(title)

    st.markdown("---")

    st.markdown("###  1. Principe")
    st.markdown("""
    Le principe du « Transfer Learning » consiste à reprendre des algorithmes pré-entraînés sur de nombreuses images en en conservant les poids.\n
    Nous complétons ensuite le modèle par quelques couches afin d’obtenir la sortie voulue, à savoir les types de document dans notre cas.\n
    """)

    st.markdown("---")

    st.markdown("###  2. Architecture")
    st.image("./assets/architecture_CV.jpg")
    with  st.expander(label='Méthodologie'):
        st.markdown("""
            * Entraînement de l'algorithme avec les poids gelés du modèle pré-entraîné.\n
            * Récupération du taux d'apprentissage du premier algorithme en fin d'apprentissage.\n
                --- FINE TUNING ---\n
            * Dégel des dernières couches du modèle pré-entraîné.\n
            * Nouvel entraînement de l'algorithme.\n
            """)
    with  st.expander(label='Callbacks utilisés'):
        st.markdown("""
            * Réduction automatique du taux d'apprentissage.
            * Arrêt automatique de l'apprentissage.
        """)

    st.markdown("---")


    st.markdown("###  3. Comparaison des modèles")
    with st.expander('Modèles testés'):
        st.markdown("""
            * VGG16 (Résolution 224x224)\n
            * VGG19 (Résolution 224x224)\n
            * ResNet50 (Résolution 224x224)\n
            * ResNet152 (Résolution 224x224)\n
            * EfficientNetB0 (Résolution 224x224)\n
            * EfficientNetB1 (Résolution 240x240)\n
            * EfficientNetB2 (Résolution 260x260)
        """)

    # Data collection
    df = pd.read_csv('../data/classification_report.csv')
    df_accuracy_target_min = df.loc[(df['index'] == "accuracy") & (df.category == "ComputerVision target_min")]
    df_accuracy_target_min = df_accuracy_target_min.loc[(df_accuracy_target_min.classifier != "ResNet50 AugmentedData")
                                                          & (df_accuracy_target_min.classifier != "EfficientNetB0 AugmentedData")]
    df_accuracy_target_min = df_accuracy_target_min.sort_values(by ='classifier', ascending = False)

    hgraph = px.scatter(df_accuracy_target_min, x = 'classifier',y = 'precision')
    st.plotly_chart(hgraph)

    with st.expander('Modèles retenu'):
        st.markdown("""
        * ResNet50
    """)

    st.markdown("---")

    st.markdown("###  4. Traitement des données")
    with st.expander(label='Augmentation des données afin de rendre le modèle plus robuste'):
        st.markdown("""
            * Rotation des images de 90 degrés,\n
            * Largeur de l'image pouvant variée de 5%,\n
            * Hauteur de l'image pouvant variée de 5%,\n
            * Image renversée horizontalement,\n
            * Zoom de 5%.\n
            """)
    with  st.expander(label="Correction du labeling des documents d'identité"):
        st.markdown("""
            * Inversion de labeling entre carte d'identité et passeport,\n
            * Supression des permis de conduire,\n
            * Suppression des images sur lesquelles apparaissent un passeport ET une carte d'identité.\n""")
    st.markdown("""=> Gain d'environ 5% sur la précision et le rappel des documents d'identité.
    """)

    st.markdown("---")

    st.markdown("###  5. Performances du ResNet50")
    st.image("./assets/Performances_ResNet50.jpg")
    st.markdown("""
        On remarque une très bonne précision sur les cartes d'identité (1.00) 
        et les passeports (0.88) avec un rappel de 0.93 sur les cartes d'identité et 0.95
        sur les passeports.
    """)

    st.markdown("---")

    st.markdown("###  6. Tester le modèle")

    uploaded_file = st.file_uploader("Choisir une image")
    if uploaded_file is not None:
        # To read file as bytes:
        image = uploaded_file.read()
        predictImage(image)
 
    
