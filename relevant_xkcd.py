# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:24:22 2020

@author: keesb
"""

import streamlit as st
import pickle
import numpy as np
import spacy


@st.cache(allow_output_mutation=True)
def load_model():
    #load Spacy languag model. Model only needs to be loaded once
    return spacy.load("en_core_web_lg")

@st.cache
def load_pickles():
    #load pickles with data which have been generated in relevant_xkcd.ipynb
    comics = pickle.load(open("comics.pickle", "rb"))
    title_matrix = pickle.load(open("title_matrix.pickle", "rb"))
    idx_to_keys = pickle.load(open("idx_to_keys.pickle", "rb"))
    
    return comics, title_matrix, idx_to_keys

def cosine_distance(a,b):
    #returns array with cosine distances between one 1d numpy array and one 2d numpy array
    return ((b - a)**2).sum(axis=1)**0.5

nlp = load_model()
comics, title_matrix, idx_to_keys = load_pickles()

st.title("relevant_xkcd")
st.text("intro text")

query = st.text_input("Enter your query")
number_of_results = st.slider('Select number of results', min_value=1, max_value=10)

#apply language model on query to get the vector attribute
doc = nlp(query)

#use the pre-loaded title embedding vectors and query embedding vector to get cosine distances
distances = cosine_distance(doc.vector, title_matrix)

#return indices of the top lowest number_of_results distances
top_candidates = (distances).argsort()[:number_of_results]

#show title, comment and image for every top candidate
for c, x in enumerate(top_candidates):
    st.header(f"Candidate {c+1}")
    st.subheader("Title: "+ (comics[idx_to_keys[x]]["title"]))
    st.write((comics[idx_to_keys[x]]["comment"]))   
    img_link = r"https:" + comics[idx_to_keys[x]]["img_link"]
    st.image(img_link, caption='relevant_xkcd', use_column_width=True)

