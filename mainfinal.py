import os
import streamlit as st
import pickle
import requests
from bs4 import BeautifulSoup
from langchain.llms.openai import OpenAI
#from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document  # Import the Document class
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Load the InstructorXL model
instructor_model = SentenceTransformer('hkunlp/instructor-xl')



# Streamlit UI
st.title("YOUR NEW RESEARCHER FRIEND")
st.sidebar.title("NEWS URLS")

# Function to fetch page content using BeautifulSoup
def fetch_page_content(url):
    try:
        response = requests.get(url) #request library jo brower kholkr content nikalti hai
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser") #.content mean usme saara word format mai aa jayega/ response.text bhi likh sakte hai
            return soup.get_text()  # Extract plain text from HTML
        else:
            st.error(f"Failed to fetch URL: {url}")
            return ""
    except Exception as e:
        st.error(f"An error occurred while fetching URL: {url}, Error: {str(e)}")
        return ""

# Function to process content into a list of Document objects
def process_urls(urls):
    documents = []
    for url in urls:
        content = fetch_page_content(url)
        if content:
            # Create a Document object for each piece of content
            documents.append(Document(page_content=content, metadata={"source": url}))
    return documents

# Sidebar input for URLs
urls = []
for i in range(3):
    url_input = st.sidebar.text_input(f"URL {i + 1}")
    if url_input:  # Ensure non-empty input is added
        urls.append(url_input)

# Button to trigger processing of URLs
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

# Initialize the OpenAI model
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    if not urls:
        st.error("Please provide at least one URL.")
    else:
        main_placeholder.text("Fetching and processing data from URLs...")

        # Fetch content from the URLs and process into Document objects
        data = process_urls(urls)

        # Check if we fetched any data
        if not data:
            st.error("No data was fetched from the provided URLs.")
        else:
            st.write(f"Data loaded. {len(data)} documents fetched.")

            # Split the documents into chunks for processing
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=200
            )
            main_placeholder.text("Splitting the documents into chunks...")
            docs = text_splitter.split_documents(data)

            if not docs:
                st.error("No chunks were created from the documents.")
            else:
                st.write(f"Splitting successful: {len(docs)} chunks created.")

                # Create embeddings using HuggingFace (OpenAIEmbeddings can be used if preferred)
                embeddings = HuggingFaceEmbeddings()
                vectorstore_hf = FAISS.from_documents(docs, embeddings)

                # Save the FAISS vector store to a file
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_hf, f)

                main_placeholder.success("Data processed and vector store saved!")

# Question input and retrieval logic
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Function to retrieve answer using HuggingFace QA pipeline
def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

query = st.text_input("Ask a question:")
file_path = "faiss_store_openai.pkl"

if query:
    try:
        if os.path.exists(file_path):

            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            retriever = vectorstore.as_retriever()
            # Retrieve only the top 3 most relevant documents
            docs = retriever.get_relevant_documents(query)[:3]


            context = " ".join([doc.page_content for doc in docs])

            answer = answer_question(query, context)

            st.header("Answer")
            st.subheader(answer)
        else:
            st.error("Vector store file not found. Please process URLs first.")

    except Exception as e:
        st.error(f"An error occurred while answering: {str(e)}")