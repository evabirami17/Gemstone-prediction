import numpy as np
import pickle
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

model1 = keras.models.load_model('model/model.h5')


def reserve(feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9):
  prediction=model1.predict([[feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9]])
  print(prediction)
  return prediction

def main():
   st.title("GemStone Prediction")

   feature1 = st.number_input("Feature 1", value=0.0)
   feature2 = st.number_input("Feature 2", value=0.0)
   feature3 = st.number_input("Feature 3", value=0.0)
   feature4 = st.number_input("Feature 4", value=0.0)
   feature5 = st.number_input("Feature 5", value=0.0)
   feature6 = st.number_input("Feature 6", value=0.0)
   feature7 = st.number_input("Feature 7", value=0.0)
   feature8 = st.number_input("Feature 8", value=0.0)
   feature9 = st.number_input("Feature 9", value=0.0)
 
   result=''
   if st.button("Predict"):
    result=reserve(feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9)
   st.success('The output is {}'.format(result))
if __name__=='__main__':
  main()