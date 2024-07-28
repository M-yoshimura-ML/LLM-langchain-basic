import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['USER_AGENT'] = os.getenv('USER_AGENT')

# Internally using BeautifulSoup4
loader = WebBaseLoader("https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")

docs = loader.load()
# print(docs)

"""
Load Data -> Docs -> Divide text into chunks -> text 
-> vectors -> vector embeddings -> vector store DB 

LLM has limitation for context size
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
print(documents[0])

faiss_index_path = "chat_models/faiss_index"

embeddings = OpenAIEmbeddings()

if os.path.exists(faiss_index_path):
    db = FAISS.load_local(folder_path=faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
else:
    # This would take time to create vector db
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(faiss_index_path)

query = "LangSmith has two usage limits: total traces and extended"
result = db.similarity_search(query)
# print(result[0].page_content)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    """
)

llm = ChatOpenAI(model="gpt-4o-mini")
document_chain = create_stuff_documents_chain(llm, prompt)
# print(document_chain)

from langchain_core.documents import Document

result2 = document_chain.invoke({
    "input": "LangSmith has two usage limits: total traces and extended",
    "context": [Document(page_content="The usage graph lets us examine how much of each usage based pricing metric we have consumed lately.")]
})

print(result2)

# Input -> Retriever --> Vectorstore DB

retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print("\n============retrieval_chain==================\n")
print(retrieval_chain)

print("\n============response==================\n")
response = retrieval_chain.invoke({
    "input": "LangSmith has two usage limits: total traces and extended"
})

print(response)
