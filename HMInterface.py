#importing the libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import hashlib,time

@st.cache(allow_output_mutation=True)
def init():
    df = pd.read_json('reductedJsonExport.json')
    return df

st.session_state["df"] = init()

if 'count' not in st.session_state:
	st.session_state.count = 0
if 'HS' not in st.session_state:
    st.session_state['HS'] = []
if 'GEN' not in st.session_state:
    st.session_state['GEN'] = []
if 'export' not in st.session_state:
    st.session_state['export'] = []

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
st.sidebar.image("logo.png")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def suggExport(HS:bool):
    if HS:
        string = "Hors-Sujet"
    else:
        string = "Formation du numérique"
    file = []
    sugg = dict()
    sugg["type"] = "relation-point",
    sugg["path"] =  [st.session_state["df"].loc[st.session_state.count,"title"],"skos:exactMatch",string]
    sugg["matchingId"] = str(hashlib.md5((sugg["path"][0] + "/" + sugg["path"][1] + "/" + sugg["path"][2]).encode('utf-8')).hexdigest())
    sugg["relationType"] = "skos:exactMatch"
    sugg["matches"] = {"source": "HSLR2",
                    "runId" : time.time(),
                    "score" : 1}
    sugg["proposed"] = "valided" 
    return sugg

def update():
    bar.progress(st.session_state.count/100)
    titre.write("<h2>" + st.session_state["df"].loc[st.session_state.count,"title"] + "</2>",unsafe_allow_html=True)
    text = ""
    if isinstance(st.session_state["df"].loc[st.session_state.count,"objectif_formation"],str):
        text += " " + st.session_state["df"].loc[st.session_state.count,"objectif_formation"]
        below.write("<h2> Objectif formation :</2>",unsafe_allow_html=True)
        below.write("<h3>" + st.session_state["df"].loc[st.session_state.count,"objectif_formation"] + "</h3>",unsafe_allow_html=True)
    if isinstance(st.session_state["df"].loc[st.session_state.count,"contenu_formation"],str):
        text += " " + st.session_state["df"].loc[st.session_state.count,"contenu_formation"]
        below.write("<h2> Contenu formation :</2>",unsafe_allow_html=True)
        below.write("<h3>" + st.session_state["df"].loc[st.session_state.count,"contenu_formation"] + "</h3>",unsafe_allow_html=True)
    if isinstance(st.session_state["df"].loc[st.session_state.count,"resultats_attendus"],str):
        text += " " + st.session_state["df"].loc[st.session_state.count,"resultats_attendus"]   
        below.write("<h2> Résultats attendus :</2>",unsafe_allow_html=True)
        below.write("<h3>" + st.session_state["df"].loc[st.session_state.count,"resultats_attendus"]   + "</h3>",unsafe_allow_html=True) 


if col2.button("Hors-Sujet"):
    st.session_state["HS"].append(st.session_state["df"].loc[st.session_state.count,"id"])
    st.session_state["export"].append(suggExport(True))
    st.session_state.count += 1


if col4.button("Formation du numérique"):
    st.session_state["GEN"].append(st.session_state["df"].loc[st.session_state.count,"id"])
    st.session_state["export"].append(suggExport(False))
    st.session_state.count += 1

    

# For newline
st.sidebar.write('\n')
st.sidebar.write('Nombre de formations classées = ', st.session_state.count)    
st.sidebar.write('Nombre de formations validées = ', len(st.session_state["GEN"]) ) 
st.sidebar.write('Nombre de formations écartées = ', len(st.session_state["HS"]) )
st.sidebar.write('Nombre de formations restantes =', len(st.session_state["df"])-st.session_state.count)    


st.sidebar.download_button(
    label="Download JSON",
    file_name="validé.json",
    mime="application/json",
    data=json.dumps({"suggestions":st.session_state["export"]}
                    ,indent=2),
)
m = st.markdown("""
<style>
#root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.css-k1vhr4.egzxvld3 > div > div:nth-child(1) > div > div.css-ocqkz7.e1tzin5v4 > div:nth-child(4) > div:nth-child(1) > div > div > div > button:hover {
    border-color: rgb(0, 255, 0);
    color: rgb(0, 255, 0);
}
#root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.css-k1vhr4.egzxvld3 > div > div:nth-child(1) > div > div.css-ocqkz7.e1tzin5v4 > div:nth-child(4) > div:nth-child(1) > div > div > div > button:active {
    background-color: rgb(0, 255, 0);
    color: rgb(255, 255, 255);
}
div.stButton > button:first-child:focus {
    border-color: rgb(225, 225, 225);
    color: rgb(0,0,0);
    box-shadow: None;
}
</style>""", unsafe_allow_html=True)

update()
