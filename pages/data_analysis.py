import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier
import time
import warnings
warnings.filterwarnings("ignore")
    
st.title(':blue[DATA ANALYSIS]')    
with st.spinner('Please Wait'):
    time.sleep(1)

selected_cols = {}
keys = []
values = []
predict_col = {}
predict_col_list = []
count = 0
predict_user_column = ""
old_cols_list = [] 
new_cols_list = []  
result = pd.DataFrame()   
train = pd.DataFrame()   
x_train = pd.DataFrame()   
x_test = pd.DataFrame()   
test = pd.Series()   
y_train = pd.Series()   
y_test = pd.Series()   
reg = LinearRegression()
all_models = []
train_cols = []
    
allentries = os.listdir(os.getcwd()+"/uploaded")
st.subheader('Select your Record which you want to Analyse:-')
select_box = st.selectbox('',allentries)
df = pd.read_csv(select_box)

st.subheader("If you want to update any values You can Update:-")
update_num = st.number_input("How many values you want to update",min_value=0)
update_num = int(update_num)
for num in range(update_num):
    num = num + 101
    old_cols_list.append(st.text_input("Current Value",key=num))
    num = num + 101
    new_cols_list.append(st.text_input("New Value",key=num))
    


df = df.replace(old_cols_list,new_cols_list)
df = df.dropna()
df.drop_duplicates(inplace = True)
result = st.experimental_data_editor(df)

st.subheader("Select What Column which You Want to Predict from Dataset:-")  
for col in result.columns:
    count = count + 1
    predict_col[col] = st.checkbox(col , key = count)



for v in predict_col.keys():
    le = len(predict_user_column)
    if le == 0: 
        if predict_col[v]:
            predict_user_column = v
            keys.append(predict_user_column)
        else: continue
    else: continue
    

        

st.subheader("Select What Columns You Want to Drop from Training Data Set")
for cols in result.columns:
    count = count + 1
    if cols == predict_user_column:
        continue
    selected_cols[cols] = st.checkbox(cols , key = count)

for key in selected_cols.keys():
    if selected_cols[key]:
        keys.append(key)
    
   
df_columns = result.columns.tolist()
keys_diff = list(set(df_columns) - set(keys))
pair_plot_values = []
pair_plot_values.append(predict_user_column)
other_plot_values = list(set(df_columns) - set(pair_plot_values))
pair_plot_values = pd.DataFrame(result.drop(other_plot_values,axis=1))
pred = st.button('predict',key = "key1")
if 'load_state' not in st.session_state:
    st.session_state.load_state = False
    
if pred or st.session_state.load_state:
    st.session_state.load_state = True
   
    st.subheader('Pair Plot:-')

    sns.pairplot(data = result[keys_diff],kind = "scatter")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()    
    
    all_models = ["Linear Regression Model"]
    selected_model = st.selectbox('Select Model for Prediction',all_models)
    
    train = result[keys_diff]
        
    if not predict_user_column:
        st.warning("Please Select Column to Predict")
        st.stop()
        
    test = result[predict_user_column]
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    
    x_train , x_test , y_train , y_test = train_test_split(train , test , test_size=0.3 , random_state = 2)
    
    if selected_model == "Linear Regression Model":
        reg = LinearRegression()
        reg.fit(x_train , y_train)

        predicted_data = reg.predict(x_test)
        
        st.title('Regression Score:-')
        plt.scatter(y_test,predicted_data)
        st.pyplot()

        reg_score = reg.score(x_test , y_test)

        title = st.text_input('Regression Score', reg_score)
        

        
    st.subheader("Please give your own Input to Predict House Price")

    user_input = {}

    for key in keys_diff:
        user_input[key] = st.text_input(key)
        
    predict_btn = st.button('predict' , key = "key2")
    
    if 'new_load_state' not in st.session_state:
        st.session_state.new_load_state = False
        
    if predict_btn or st.session_state.new_load_state:
        st.session_state.new_load_state = True
        for u in user_input.keys():
            if user_input[u] == "":
                st.warning('This is a warning', icon="⚠️")
                break
            
            else:   
                user_input = pd.DataFrame(user_input,index=[0])
                new_predict = reg.predict(user_input)

                st.subheader("Predicted House Price:")
                st.text_input("Predicted House Price",new_predict)
                st.balloons()
                st.stop()

            
    
  
    
   
        
    
    
    
   
    

        