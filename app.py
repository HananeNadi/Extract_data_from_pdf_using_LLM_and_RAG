from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import uuid

from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM using Google Generative AI
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
response = llm.invoke("Tell me a general joke.")

##Process the pdf document
loader=PyPDFLoader("data\Oppenheimer-2006-Applied_Cognitive_Psychology.pdf")
data=loader.load()
# print(len(data)) //number of pages=3


##split the document
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200,length_function=len,separators=["\n\n","\n"," "])
chunks=text_splitter.split_documents(data)
# print(len(chunks))//11 chunks

##create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")
test_vector = embeddings.embed_query("hello","world!")
# print(test_vector[:3])


def create_vectorDB(chunks, embeddings, vectorstore_path):

    ##to ensure that we don't have duplicate vectors
    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []
    
    unique_chunks = [] 
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk) 

    # Use 'embedding_function' instead of 'embedding'
    vectorDB = Chroma.from_documents(documents=unique_chunks, 
                                    ids=list(unique_ids),
                                    embedding=embeddings, 
                                    persist_directory=vectorstore_path)

    vectorDB.persist()
    
    return vectorDB

# Call the function with the correct parameter
vectorstore = create_vectorDB(chunks=chunks, 
                            embeddings=embeddings, 
                            vectorstore_path="vectorstore_test")




#Query for relevant data
##Load the vectorstore
vectorstore= Chroma(persist_directory="vectorstore_chroma", embedding=embeddings)

###Create retriever and get relevant chunks
retriever=vectorstore.as_retriever(search_type="similarity"), #Cos distance
retrieved_docs = retriever.invoke("What is the title of the paper?")
print(len(retrieved_docs))




