import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
import pathlib


current_dir = os.getcwd()
file_name = st.file_uploader('Select file')


if file_name:
    if file_name.type=="text/csv":
        filepath = current_dir +"/"+file_name.name
        df = pd.read_csv(filepath)
        targetpath = current_dir +"/"+"uploaded/"+file_name.name
        shutil.copyfile(filepath,targetpath)
        df
        st.success("File Uploaded Successfully")
        st.markdown('<a href="/data_analysis" target="_self"><button>ANALYSE DATA</button></a>', unsafe_allow_html=True)  
    else:
        st.warning("upload csv file only!!!!!")
     
    
    
