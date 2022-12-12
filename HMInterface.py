#importing the libraries
import streamlit as st
import streamlit.components.v1 as components
import pickle
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer


@st.cache(allow_output_mutation=True)
def init():
    model = SentenceTransformer('Sahajtomar/french_semantic')
    lr = pickle.load(open('pythonBert/data/models/HSLR2.sav', 'rb'))
    return model,lr

model,lr = init()

if 'count' not in st.session_state:
	st.session_state.count = 0
if 'HS' not in st.session_state:
    st.session_state['HS'] = []
if 'GEN' not in st.session_state:
    st.session_state['GEN'] = []

above = st.container()
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
below = st.container()

# Designing the interface
above.title("Classification des formations GEN")
# For newline
above.write('\n')
bar = above.progress(0)
titre = below.empty()
description = below.empty()
prediction = col3.empty()
st.sidebar.image("pythonBert/Visualization/logo.png")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['xlsx'] )
with open('pythonBert/Visualization/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



def classify(i):
    texte = df.loc[i,"text"][:2000]
    try:
        proba = lr.predict_proba(model.encode(texte,convert_to_tensor=True).reshape(1, -1))[0][1]
    except:
        try:
            proba = lr.predict_proba(model.encode(texte[:1000],convert_to_tensor=True).reshape(1, -1))[0][1]
        except:
            proba = np.nan()
    return proba

def reset():
    st.session_state.count = 0
    #A finir


def update():
    if st.session_state.count == 1000:
        reset()
    bar.progress(st.session_state.count/100)
    titre.write("<h2>" + df.loc[st.session_state.count,"results.title"] + "</2>",unsafe_allow_html=True)
    description.write("<h3>" + df.loc[st.session_state.count,"text"] + "</h3>",unsafe_allow_html=True)
    proba = classify(st.session_state.count)
    prediction.write("<h4>" + str('%.1f'%(proba*100)) + "%" + "</h4>",unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def openFile():
    return pd.read_excel(uploaded_file).loc[:99]

if uploaded_file is not None:
    df = openFile()
    if st.session_state.count == 0:   
        update()      




if col2.button("Hors-Sujet"):

    if uploaded_file is None:
        
        description.table(pd.DataFrame([["Uploadez d'abord le dataset"]]))
    
    else:
        st.session_state["HS"].append(df.loc[st.session_state.count,"results.id"])
        st.session_state.count += 1
        update()

if col4.button("Formation GEN"):

    if uploaded_file is None:
        
        titre.table(pd.DataFrame([["Uploadez d'abord le dataset"]]))

    else:
        st.session_state["GEN"].append(df.loc[st.session_state.count,"results.id"])
        st.session_state.count += 1
        update()
        
# For newline
st.sidebar.write('\n')
st.sidebar.write('Nombre de formations classées = ', st.session_state.count)    
st.sidebar.write('Nombre de formations validées = ', len(st.session_state["GEN"]) ) 
st.sidebar.write('Nombre de formations validées = ', len(st.session_state["HS"]) )
if uploaded_file is not None:
    st.sidebar.write('Nombre de formations restantes =', len(df)-st.session_state.count)    


st.sidebar.download_button(
    label="Download JSON",
    file_name="validé.json",
    mime="application/json",
    data=json.dumps({"HS":st.session_state["HS"],
                    "GEN":st.session_state["GEN"]}
                    ,indent=2),
)


