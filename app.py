import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np


with open('preprocessor.pkl','rb')as f:
    preprocessor=pickle.load(f)

model=load_model('model.keras')
df = pd.read_csv('processed_dataset.csv')

st.title('Money Laundering Detection')

col1,col2=st.columns(2)

with col1:
    From_Bank=st.selectbox('From Bank',df['From Bank'].unique())

with col2:
    To_Bank=st.selectbox('To Bank',df['To Bank'].unique())


Amount_Paid=st.number_input('Enter the Amount to Pay',min_value=100)
Payment_Currency=st.selectbox('Select Currency',df['Payment Currency'].unique())
Payment_Format=st.selectbox('Select the Payment_Format',df['Payment Format'].unique())
Hour=st.selectbox('Enter the Hour of Payment  Done',df['Hour'].unique())
Dayofweek=st.selectbox('Enter Day of week',df['Dayofweek'].unique())


data=pd.DataFrame({'From Bank':[From_Bank],'To Bank':[To_Bank],'Amount Paid':[Amount_Paid],
                   'Payment Currency':[Payment_Currency],'Payment Format':[Payment_Format],
                   'Hour':[Hour],'Dayofweek':[Dayofweek]})


data['From Bank'] = data['From Bank'].astype(object)
data['To Bank'] = data['To Bank'].astype(object)
data['Payment Currency'] = data['Payment Currency'].astype(object)
data['Payment Format'] = data['Payment Format'].astype(object)


if st.button('Predict'):
    preprocessed_data=preprocessor.transform(data)
    ypred=np.where(model.predict(preprocessed_data)>0.5,1,0)

    if ypred[0][0]==1:
       st.warning('Money is Laundering')
    else:
        st.success('No Money Laundering')
    prob = model.predict(preprocessed_data)
   
    st.success(f"Probability Of Fraud Transaction: {prob[0][0]:.2f}")
