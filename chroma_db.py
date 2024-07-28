"""
Chroma is a AI-native open-source vector database focused on developer productivity
 and happiness. Chroma is licensed under Apache 2.0.

https://python.langchain.com/v0.2/docs/integrations/vectorstores/
"""
# pip install chromadb

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = TextLoader("documents/speech.txt")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

embeddings = OllamaEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)

# query it
query = "What does the speaker believe is the main reason the United States should enter the war?"
docs = vectordb.similarity_search(query)
print(docs[0].page_content)

# saving to disk
vectordb = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")


# load from disk
db2 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
docs = db2.similarity_search(query)
print(docs[0].page_content)


# similarity Search With Score
docs = vectordb.similarity_search_with_score(query)
print(docs)



