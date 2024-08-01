# Arxix - Research
# Tools creation
import os

from dotenv import load_dotenv
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
import openai


load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
print(wiki.name)

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
print(arxiv.name)

tools = [wiki, arxiv]

# Custom tools[RAG Tool]
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()

print(retriever)

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(retriever, "langsmith-search", "Search any information about LangSmith.")

print(retriever_tool.name)

tools = [wiki, arxiv, retriever_tool]
print(tools)

# Run all this tools with Agents and LLM Models
# Tools, LLM -> AgentExecutor
from langchain_groq import ChatGroq

groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt Template
from langchain import hub
# https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_functions_agent/
prompt = hub.pull("hwchase17/openai-functions-agent")

# print(prompt.messages)

# Agent
from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(llm, tools, prompt)

# print(agent)

# Agent Executor
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor)

# result = agent_executor.invoke({"input": "Tell me about Langsmith"})
# result = agent_executor.invoke({"input": "What is Machine Learning"})

# https://arxiv.org/abs/1706.03762
result = agent_executor.invoke({"input": "What's the paper 1706.03762 about ?"})
print(result)
