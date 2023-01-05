#importing the libraries
import streamlit as st
import pickle
import pandas as pd
import hashlib
import time
import json
from sentence_transformers import SentenceTransformer
st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
def init():
    model = SentenceTransformer('Sahajtomar/french_semantic')
    lr2 = pickle.load(open('matchingLR2.sav', 'rb'))
    families = json.load(open(r'families.json',encoding="utf-8"))
    return model,lr2,families

model,lr2,families = init()

if 'count' not in st.session_state:
	st.session_state.count = -1
if 'metiers' not in st.session_state:
    st.session_state['metiers'] = []


above = st.container()
below = st.container()
col1, col2 = st.columns([2,1])

# Designing the interface
above.title("Classification des formations GEN")
# For newline
above.write('\n')
bar = above.progress(0)
titre = below.empty()
description = col2.empty()
st.sidebar.image("logo.png")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['json'] )
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def suggExport():
    file = []
    sugg = dict()
    sugg["type"] = "relation-point",
    sugg["path"] =  [st.session_state["df"].loc[st.session_state.count,"title"],"skos:exactMatch",st.session_state["selected_option"].split("-")[0][:-1]]
    sugg["matchingId"] = str(hashlib.md5((sugg["path"][0] + "/" + sugg["path"][1] + "/" + sugg["path"][2]).encode('utf-8')).hexdigest())
    sugg["relationType"] = "skos:exactMatch"
    sugg["matches"] = {"source": "matchingLR2",
                    "runId" : time.time(),
                    "score" : st.session_state["selected_option"].split("-")[1][1:-2]}
    sugg["proposed"] = "valided" 
    return sugg


@st.cache(allow_output_mutation=True)
def calcul(text):
    texte = text[:1000]
    try:
        proba2 = lr2.predict_proba(model.encode(texte,convert_to_tensor=True).reshape(1,-1))
        rounded_percentages = [round(x, 3)*100 for x in proba2[0]]
        options = sorted(zip(rounded_percentages, families.keys()), reverse=True)
    except:
        try:
            proba2 = lr2.predict_proba(model.encode(texte[:1000],convert_to_tensor=True).reshape(1,-1))
            rounded_percentages = [round(x, 3)*100 for x in proba2[0]]
            options = sorted(zip(rounded_percentages, families.keys()), reverse=True)
        except:
            proba2 = [0] * len(families.keys())
            rounded_percentages = [round(x, 3)*100 for x in proba2]
            options = sorted(zip(rounded_percentages, families.keys()), reverse=True)
    options = ["---"]+[t[1]+ " - " + str(t[0]) + " %" for t in options]
    return options

def addToList():
    st.session_state["metiers"].append(suggExport())

def update():
    bar.progress(st.session_state.count/100)
    titre.write("<h2>" + st.session_state["df"].loc[st.session_state.count,"title"] + "</2>",unsafe_allow_html=True)
    text = ""
    if isinstance(st.session_state["df"].loc[st.session_state.count,"objectif_formation"],str):
        text += " " + st.session_state["df"].loc[st.session_state.count,"objectif_formation"]
        col1.write("<h2> Objectif formation :</2>",unsafe_allow_html=True)
        col1.write("<h3>" + st.session_state["df"].loc[st.session_state.count,"objectif_formation"] + "</h3>",unsafe_allow_html=True)
    if isinstance(st.session_state["df"].loc[st.session_state.count,"contenu_formation"],str):
        text += " " + st.session_state["df"].loc[st.session_state.count,"contenu_formation"]
        col1.write("<h2> Contenu formation :</2>",unsafe_allow_html=True)
        col1.write("<h3>" + st.session_state["df"].loc[st.session_state.count,"contenu_formation"] + "</h3>",unsafe_allow_html=True)
    if isinstance(st.session_state["df"].loc[st.session_state.count,"resultats_attendus"],str):
        text += " " + st.session_state["df"].loc[st.session_state.count,"resultats_attendus"]   
        col1.write("<h2> Résultats attendus :</2>",unsafe_allow_html=True)
        col1.write("<h3>" + st.session_state["df"].loc[st.session_state.count,"resultats_attendus"]   + "</h3>",unsafe_allow_html=True) 
    with col2:
	options = calcul(text)
    col2.selectbox("Sélectionnez le métier",options,on_change=addToList,key="selected_option")

@st.cache(allow_output_mutation=True)
def openFile():
    return pd.read_json(uploaded_file).loc[:99]


if col2.button("Retour"):
    if st.session_state.count > 0:
        del st.session_state["metiers"][-1]
        st.session_state.count -= 2
    else:
        st.session_state.count -= 1

if uploaded_file is not None:
    st.session_state["df"] = openFile() 
    st.session_state.count +=1
    update()      

    
        
# For newline
st.sidebar.write('\n')
st.sidebar.write('Nombre de formations classées = ', st.session_state.count)    
st.sidebar.write('Nombre de formations validées = ', len(st.session_state["metiers"]))
if uploaded_file is not None:
    st.sidebar.write('Nombre de formations restantes =', len(st.session_state["df"])-st.session_state.count)    


st.sidebar.download_button(
    label="Download JSON",
    file_name="validé.json",
    mime="application/json",
    data=json.dumps({"suggestions":st.session_state["metiers"]}
                    ,indent=2),
)


