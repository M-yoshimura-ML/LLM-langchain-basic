# pip install fastapi uvicorn
# pip install langserve
"""
LangServe:
helps developers deploy LangChain runnables and chains as a REST API.
you can see API docs with : http://localhost:8000/docs

https://python.langchain.com/v0.2/docs/langserve/
"""
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os

load_dotenv()

groq_api_key = os.environ.get('GROQ_API_KEY')
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# 1. Create prompt template
system_template = "Template the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

parser = StrOutputParser()

# create chain
chain = prompt_template | model | parser

# App definition
app = FastAPI(title="LangChain",
              version="1.0",
              description="A simple API server using LangChain runnable interfaces")

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

