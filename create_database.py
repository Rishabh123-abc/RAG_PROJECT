#Load pdf
#split into chunks
#create the embeddings
#store into chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
url = 'https://www.geeksforgeeks.org/python/reading-writing-text-files-python/'
data = WebBaseLoader(url)
docs = data.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory='chroma_db'
)
result = vectorstore.similarity_search('What are the modes in file?,k=2')
for r in result:
    print(r.page_content)
    print(r.metadata)
retriever = vectorstore.as_retriever()
docs = retriever.invoke('What are read mode in file')
for d in docs:
    print(d.page_content)