import streamlit as st
import pandas as pd

st.title('💬 Summary')
if "summary" not in st.session_state:
    st.session_state["summary"] = ""

import os
a='D:\Dallas AI\Clean Transcripts'
a=os.listdir(a)
a=[i.replace('.txt','') for i in a]
a.insert(0,' ')

if conid := st.selectbox(
    "Conversational ID",
    a,):
    if conid !=' ':
    
        df=pd.read_csv('D:\Dallas AI\summaries.csv')
        st.session_state.summary=df.loc[df['File Name']==conid]['Summary'].values[0]
    else:
            st.session_state.summary=''

st.write(st.session_state["summary"])