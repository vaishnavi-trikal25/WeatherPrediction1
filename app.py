#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
Created on Mon Feb 7 02:02:49 2022

@auther: Vaishnavi Trikal

"""
import numpy as np
import pickle
import pandas as pd


import streamlit as st 

from PIL import Image


pickle_in = open("Seattle_WeatherPredicton.csv","rb")
Seattle_WeatherPredicton.csv=pickle.load(pickle_in)


def welcome():
    return "Welcome All"

#@app.route
def predict_WeatherPrediction(date,precipitation,temp_max,temp_min,wind_speed,weather):
 prediction=classifier.predict([[date,precipitation,temp_max,temp_min,wind_speed,weather]])
 print(prediction)
 return prediction



def main():
    st.title("")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit WeatherPrediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    date = st.text_input("date","Type Here")
    precipitation = st.text_input("precipitation","Type Here")
    temp_max = st.text_input("temp_max","Type Here")
    temp_min = st.text_input("temp_min","Type Here")
    weather = st.text_input("weather","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_WeatherPrediction(date,precipitation,temp_max,temp_min,weather)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()


# In[ ]:




