import os.path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

"""
Facebook AI Similarity Search(Faiss) is a library for efficient similarity search and clustering
of dense vectors. it contains algorithms that search in sets of vectors of any size, up to ones that
possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.
"""

loader = TextLoader("documents/speech.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
docs = text_splitter.split_documents(documents)

# print(docs)

faiss_index_path = "faiss_index"

# this program requires llama2 in your local. run below command in your local.
# >ollama run llama
embeddings = OllamaEmbeddings()

if os.path.exists(faiss_index_path):
    db = FAISS.load_local(folder_path=faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
else:
    # This would take time to create vector db
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(faiss_index_path)

# print(db)

# query = "What does the speaker believe is the main reason the United States should enter the war ?"
query = "How does the speaker describe the desired outcome of the war?"
docs = db.similarity_search(query)

print(docs[0])

print("\n=============================================================\n")

# Retriever
"""
We can also convert the vectorstore into a Retriever class. 
This allows us to easily use it in other LangChain methods, 
which largely work with retrievers
"""

retriever = db.as_retriever()
docs = retriever.invoke(query)
print(docs[0].page_content)

print("\n=============================================================\n")
# Similarity Search with Score
docs_and_score = db.similarity_search_with_score(query)
print(docs_and_score)

print("\n=============================================================\n")
embedding_vector = embeddings.embed_query(query)

# print(embedding_vector)

docs_score = db.similarity_search_by_vector(embedding_vector)
print(docs_score)
