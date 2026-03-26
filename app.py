import pickle as pk
import streamlit as st
import numpy as np

model=pk.load(open("model.pkl",'rb'))
g=st.number_input("Glucose")
b=st.number_input("bloodpressure")
if st.button("status"):
   features=[[g,b]]
   result=model.predict(features)
   if(result[0]==1):
       st.success("yes")
   else:
       st.success("no")    