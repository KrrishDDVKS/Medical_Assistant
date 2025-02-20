import os
from dotenv import load_dotenv,dotenv_values
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
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

embed = OpenAIEmbeddings(
model='text-embedding-ada-002',
openai_api_key=os.environ.get('OPEN_API_KEY')
)


vectorstore = PineconeVectorStore(index_name=index_name, embedding=embed)

prompt_template='''You are a medical assistant that accepts only medical-related input. Accept the user’s symptoms as input and retrieve probable diseases, diagnoses, and prescriptions from a vector database. If the input is irrelevant to the medical field, politely inform the user that the data is insufficient to provide a diagnosis. Ensure responses are strictly based on the metadata retrieved from the vector database, without making assumptions or providing opinions outside of it.    
    Text:
    {context}'''
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context"]
)

retriever = VectorStoreRetriever(vectorstore=vectorstore)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'],
                   model_name='gpt-4o',
                   temperature=0.0),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},)
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
    answer = qa_chain.run(query=prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
