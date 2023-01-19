import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from collections import Counter


title = "Visualisation des données"
sidebar_name = "Visualisation des données"

def run():
    st.title(title)

    df = pd.read_csv('../data/data.csv')
     
    st.markdown("##  Distribution par type de document")
  
    htype = px.histogram(df,y="type", barmode = "group",height=700,width=800)
    st.plotly_chart(htype)

    st.markdown("##  Distribution par orientation")
    st.markdown("")
    hland = px.histogram(df,y='type',color='landscape',barmode = "group",height=700,width=800)
    st.plotly_chart(hland)

    st.markdown("##  Distribution par langue")
    st.markdown("On peut constater que les deux langues les plus utilisées dans notre jeu des données sont les langues françaises et anglaises.")
    hlang = px.histogram(df,x="lang_code",color='lang_code')
    st.plotly_chart(hlang)

   
    st.markdown("##  Le mot les plus utilisées")
    all_text = ' '.join(str(i).lower() for i in df[df['text']!='']['text'])
    dico = Counter(all_text.split())
    data = {'word' : [m[0] for m in dico.most_common(15)],
                    'freq' : [m[1] for m in dico.most_common(15)]}
    word_df = pd.DataFrame(data)
    hword = px.histogram(word_df,x="freq",y="word")
    st.plotly_chart(hword)
    
    st.markdown("##  Mots les plus utilisés par type de document")
    r,c=0,0
    word_df = pd.DataFrame(columns=['word','freq','type'])
    for dtype in df['type'].unique():
        all_text = ' '.join(str(i).lower() for i in df[(df['type']==dtype) & (df['text']!='')].text)
        dico = Counter(all_text.split())
        if (len(dico.most_common(10)) > 0):
            data = {'word' : [m[0] for m in dico.most_common(5)],
                    'freq' : [m[1] for m in dico.most_common(5)],
                    'type' : [dtype for m in dico.most_common(5)]}
            word_df = word_df.append(pd.DataFrame(data),ignore_index=True)
            c+=1
            if(c==3):c=0;r+=1

    hwordt = px.bar(word_df,x="freq",y="word",orientation='h',color='type',height=700,width=800)
    st.plotly_chart(hwordt)


    """
    A COMPLETER
    """