from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pydantic import BaseModel, Field  # Pydantic v2



import uuid
import pandas as pd

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


def create_vectorstore(chunks, embeddings, vectorstore_path):

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

    # Create a new Chroma database from the documents
    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        ids=list(unique_ids),
                                        embedding=embeddings, 
                                        persist_directory = vectorstore_path)
    
    return vectorstore

# Call the function with the correct parameter
# Create vectorstore
vectorstore = create_vectorstore(chunks=chunks, 
                                embeddings=embeddings, 
                                vectorstore_path="vectorstore_chroma")




#Query for relevant data
#Load the vectorstore
vectorstore = Chroma(persist_directory="vectorstore_chroma", embedding_function=embeddings)

###Create retriever and get relevant chunks
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
retriever_docs = retriever.invoke("What is the title of the paper?")
# print(retriever_docs)

# Prompt template

prompt_system = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
""")

##concatenate context text 
context_text = "\n\n---\n\n".join([doc.page_content for doc in retriever_docs])


# prompt = prompt_system.format(context=context_text, 
#                                 question="What is the title of the paper?")
# print(llm.invoke(prompt))


#Using Langchain Expression Language LCEL
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create stuff documents chain using LCEL.
#
# This is called a chain because you are chaining together different elements
# with the LLM. In the following example, to create the stuff chain, you will
# combine the relevant context from the website data matching the question, the
# LLM model, and the output parser together like a chain using LCEL.
#
# The chain implements the following pipeline:
# 1. Extract the website data relevant to the question from the Chroma
#    vector store and save it to the variable `context`.
# 2. `RunnablePassthrough` option to provide `question` when invoking
#    the chain.
# 3. The `context` and `question` are then passed to the prompt where they
#    are populated in the respective variables.
# 4. This prompt is then passed to the LLM (`gemini-pro`).
# 5. Output from the LLM is passed through an output parser
#    to structure the model's response.
rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_system | llm
        )
# print(rag_chain.invoke("What's the title of this paper?"))


# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# response = rag_chain.invoke({"input": "what is the title of the paper?"})
# print(response["answer"])


class ExtractedInfo(BaseModel):
    """Extracted information about the research article"""
    paper_title:str =Field(description="Title of the paper") 
    publication_year:str =Field(description="Year of publication of the paper") 
    paper_authors: str =Field(description="Authors of the paper") 
    paper_summary:str =Field(description="Summary of the paper") 



rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_system | llm.with_structured_output(ExtractedInfo)
        )


# Invoke the chain to get the structured output
extracted_infos = rag_chain.invoke("What's the title,publication year ,authors and summary")

# Convert the response to a dictionary (Pydantic models can be converted to dict easily)
extracted_infos_dict = extracted_infos.model_dump()

# Convert the dictionary to a Pandas DataFrame
df = pd.DataFrame([extracted_infos_dict])

df.to_csv("data/extracted_info.csv", index=False)


# Print or inspect the DataFrame
# print(df)
