import os
import re
from dotenv import load_dotenv,dotenv_values
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
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

llm=ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'],
                   model_name='gpt-4o',
                   temperature=0.0)

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embed)

prompt_template='''If Medical Symptoms type yes else give politely inform the user that the data is insufficient to provide a diagnosis   
    Text:
    {context}'''
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context"]
)

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
    chain = LLMChain(llm=llm, prompt=PROMPT)
    answer=chain.run(prompt)
    if re.search(r'\bYes\b', answer):
        prompt_template='''Accept the user’s symptoms as input and provide probable diseases, diagnoses and prescription using only the information stored in the vector database. politely inform the user that the data is insufficient to provide a diagnosis when the given prompt is not relavent to Medical Symptoms.    
        Text:
        {context}'''
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context"]
        )

        retriever = VectorStoreRetriever(vectorstore=vectorstore)
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},)

        answer = qa_chain.run(query=prompt)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
        
    else:
        prompt_template='''Accept the queries as a customer care and generate an accurate reply.   
            Text:
            {context}'''
        PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context"])
        chain = LLMChain(llm=llm, prompt=PROMPT).run(prompt)
        st.session_state.messages.append({"role": "assistant", "content": chain})
        st.chat_message("assistant").write(chain)
        st.chat_message("assistant").write("This is answered by second Agent. The Main purpose of this app is to detect disease from symptom. Please provide the Symptom")
