import os
import re
from dotenv import load_dotenv,dotenv_values
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
import streamlit as st

st.set_page_config(
    page_title="MediConnect AI",
    page_icon=":hospital:",
    layout="wide",
)
config = dotenv_values("keys.env")
os.environ['OPENAI_API_KEY'] = st.secrets["OPEN_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINE_CONE_KEY"]

index_name = "disease-symptoms-gpt-4"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(index_name)

client=OpenAI(api_key=os.environ['OPENAI_API_KEY'])
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding
    
llm=ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'],
                   model_name='gpt-4o',
                   temperature=0.0)


st.markdown("<h1 style='text-align: center; color: black;'>MediConnect AI 🏥</h1>", unsafe_allow_html=True)
st.header("Symptom-Based Diagnosis :mask:")
st.write(
        '''
    <b>Objective:</b> Assist patients in understanding potential diagnoses based on their reported symptoms.<br>
    <b>Details:</b> Utilize past conversation data to match symptoms with possible diagnoses, offering preliminary insights and advice.<br>
    <b>Benefit:</b> Empowers patients with information, aiding in early detection and preparation before formal consultations with doctors.</span>''',
        unsafe_allow_html=True,)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! MediConnect AI is here to help you diagnose symptoms. How can I assist you today?"
}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    messages=[SystemMessage(content="If Medical Symptoms and related to body parts type yes else give politely inform the user that the data is insufficient to provide a diagnosis"),
                          HumanMessage(content=prompt)]
    chat_response = llm.invoke(messages)
    answer=chat_response.content

    if re.search(r'\bYes\b', answer):
    
        query_vec = embed(prompt)

        results = index.query(
            vector=query_vec,
            top_k=3,
            include_metadata=True
        )

        texts = [match["metadata"]["text"] for match in results["matches"]]

        
        
        prompt='''Accept the user’s symptoms as input and provide as output the probable diseases, diagnoses and prescription using only the information stored in the vector database. politely inform the user that the data is insufficient to provide a diagnosis when the given prompt is not relavent to Medical Symptoms.    
        Symptoms:
        {texts}
        Disease:'''

        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0)
        
        answer=response.choices[0].message.content
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
        
    else:
        
        messages=[SystemMessage(content="Accept the queries as a customer care and generate an accurate reply."),
                          HumanMessage(content=prompt)]
        chat_response = llm.invoke(messages)
        answer=chat_response.content
        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.chat_message("assistant").write(answer)
        st.chat_message("assistant").write("This is answered by second Agent. The Main purpose of this app is to detect disease from symptom. Please provide the Symptom")












