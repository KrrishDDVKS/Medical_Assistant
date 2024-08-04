import os
from dotenv import load_dotenv,dotenv_values
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
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



retriever = VectorStoreRetriever(vectorstore=vectorstore)
qa_chain = RetrievalQA.from_chain_type(
llm=ChatOpenAI(api_key=os.environ.get("OPEN_API_KEY"),model_name='gpt-4o',
temperature=0.0),
chain_type="stuff",
retriever=retriever
)


st.title("Symptom Checker :mask:")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    answer = qa_chain.run(query=prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
