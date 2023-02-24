#importing the libraries
import streamlit as st
import pickle
import pandas as pd
from keras.models import model_from_json
import numpy as np
import torch
from keras.layers import Dense
from keras import Sequential
st.set_page_config(page_title="Classification GEN",
                   page_icon="src/Visualization/iconGEN.ico",
                   layout="wide",
    )


@st.cache_resource(ttl=3600)
def init():
    families = ['Traffic Management', 'Test', 'Télécoms', 'Système et réseaux', "Système d'information", 'Support technique', 'Sécurité IT', 'SEA/SEO/SEM', 'Marketing digital', 'Management Produit', 'Logiciel', 'Jeux vidéos', 'IoT / Robotique', 'Impression 3D', 'IA / Machine learning', 'Growth hacking', 'Graphisme / Design', 'Gestion de site web', 'Gestion de projet digital', 'Formation au numérique', 'Entrepreneuriat', 'e-commerce', 'Données personnelles', 'Digital business Development', 'DevOps', 'Développement', 'Data / Big data', 'Cybersécurité', 'Création de vidéos', 'Content management / rédaction web', 'Community management', 'Communication digitale', 'Cloud', 'Blockchain', 'Administration de base de données', 'CRM']
    model = model_from_json(open("confirmed.json").read())
    model.load_weights("confirmed_w.h5")
    df = pd.read_excel("currentExport.xlsx")
    liste = pd.read_json("familyTagging.json")
    tensors = torch.load("tensors.pt")
    HS = pickle.load(open("crossedOut","rb"))
    OGlength = len(liste)
    return families,model,df,liste,tensors,HS,OGlength

families,model,st.session_state['df'],st.session_state['métiers'],tensors,st.session_state.HS,st.session_state.OGL = init()
st.session_state.count = np.random.randint(len(st.session_state.df))

above = st.container()
below = st.container()
col1, col2 = st.columns([2,1])
above.title("Classification des formations GEN")
above.write('\n')
titre = below.empty()
description = col2.empty()

st.sidebar.image("src/Visualization/logoGEN.png")

with open('src/Visualization/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def addToList():
    row = len(st.session_state["métiers"])
    st.session_state["métiers"].loc[row,"id"] = st.session_state["df"].loc[st.session_state.count,"id"] 
    familles = []
    for _,famille in st.session_state["selected_options"]:
        familles.append(famille)
    st.session_state["métiers"].at[row,"familles"] =  familles

def crossOut():
    st.session_state.HS.append(st.session_state["df"].loc[st.session_state.count,"id"] )

def update():
    titre.write("<h2>" + st.session_state["df"].loc[st.session_state.count,"title"] + "</2>",unsafe_allow_html=True)
    text = ""
    if isinstance(st.session_state["df"].loc[st.session_state.count,"objectif_formation"],str):
        text += " " + st.session_state["df"].loc[st.session_state.count,"objectif_formation"]
        col1.write("<h3> Objectif formation :</h3>",unsafe_allow_html=True)
        col1.write(st.session_state["df"].loc[st.session_state.count,"objectif_formation"],unsafe_allow_html=True)
    with col2:
        with st.form("classification"):
            left,center,right = st.columns([1,1,1])
            preds = model(tensors[st.session_state.df.loc[st.session_state.count,"id"]].reshape((1,1024)).numpy() ,training=False).numpy()
            familiesSorted = sorted(zip(preds[0], families), reverse=True)
            st.multiselect("Sélectionnez les métier",familiesSorted,key="selected_options",default=[familiesSorted[0]],max_selections=3,format_func=lambda x: x[1]+ " - " + str('%.1f'%(100*x[0])) + "%")
            left.form_submit_button("Valider",on_click=addToList)
            center.form_submit_button("Next")
            right.form_submit_button("Hors-Sujet",on_click=crossOut)
        
        left,center,right = st.columns([1,1,1])
        left.metric("Heures :",st.session_state.df.loc[st.session_state.count,"learningTime___hours"])
        right.metric("Mois :",st.session_state.df.loc[st.session_state.count,"monthsDuration"][1:-1])
        center.write(st.session_state.df.loc[st.session_state.count,"aggregatorProvider"])



update()        

       
# For newline
st.sidebar.write('\n')
st.sidebar.metric('Nombre de classées', len(st.session_state["métiers"]),delta=len(st.session_state.métiers)-st.session_state.OGL)    

if st.sidebar.button("Mettre à jour le modèle"):
    with st.sidebar:
        with st.spinner("Mise à jour"):
            tensors = torch.load("data/tensors/tensors.pt")
            df = pd.read_json("data/jsons/familyTagging.json")


            X = []
            y = np.zeros((len(df),len(families)))
            for rank,row in df.iterrows():
                X.append(tensors[row["id"]])
                for famille in row["familles"]:
                    y[rank,families.index(famille)] = 1

            X = torch.stack(X)
            X = X.numpy()
            print(y)


            model = Sequential()
            model.add(Dense(512, activation='relu', input_dim=1024,input_shape=(1024,)))
            model.add(Dense(256, activation='sigmoid'))
            model.add(Dense(128, activation='sigmoid'))
            model.add(Dense(36, activation='sigmoid'))
            # Compile the model
            model.compile(optimizer='adam', 
                        loss='binary_crossentropy', 
                        metrics=['accuracy'])

            model.fit(X, y, epochs=100)

            with open("data/models/confirmed.json","w") as file:
                file.write(model.to_json())
            model.save_weights("data/models/confirmed_w.h5")

if st.sidebar.button("Sauvegarder"):
    st.session_state["métiers"].to_json("familyTagging.json",orient="records")
    pickle.dump(st.session_state.HS,open("crossedOut","wb"))
