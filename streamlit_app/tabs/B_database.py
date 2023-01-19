import streamlit as st
import pandas as pd


title = "Présentation des données"
sidebar_name = "Présentation des données"


def run():
    st.title(title)

    df = pd.read_csv('../data/data_with_meta.csv')
    st.markdown("Le jeu de données comporte " + str(df.shape[0]) + " images labellisées selon 22 types.")
    
   
    st.markdown("---")
    st.dataframe(data=df.head(10))
    if st.button(label="View sample"):
        row = df.sample()
        st.markdown("###  " + row['type'].values[0])
        st.image('../data/final/' + row['filename'].values[0])

        if st.button(label="Hide"):
            st.markdown("Hide")
