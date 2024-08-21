import os
from io import BytesIO
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

import streamlit as st

# Load environment variables
load_dotenv(".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def process_pdf_and_ask_question(pdf_file, question):
    # Load PDF document
    loader = PyPDFLoader(pdf_file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    split_data = text_splitter.split_documents(data)
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    split_data = text_splitter.split_documents(data)
    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectorstore = Chroma.from_documents(documents=split_data, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":10})

    # Set up LLM and prompts
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_output_tokens=None, timeout=None, google_api_key=GOOGLE_API_KEY)
    system_prompt = (
        """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, say that you don't know.
        
        {context}
        """
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "(input)")
        ]
    )

    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    reg_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get response
    response = reg_chain.invoke({"input": question})
    return response['answer']

# Streamlit UI
st.title('Research Question Answering')

uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
st.write()
question = st.text_input("Enter your question about the PDF")
if st.button('Get Answer'):
    if uploaded_pdf and question:
         pdf=uploaded_pdf.name
         answer = process_pdf_and_ask_question(pdf, question)
         st.text_area("Answer", answer, height=300)
    else:
         st.error("Please upload a PDF and enter a question.")
