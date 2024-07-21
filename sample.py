import os

from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


load_dotenv()

template = PromptTemplate(template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""", input_variables=["fruit"])


# llm = OpenAI(openai_api_key="sk-...")
llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model="gpt-4o-mini",
    temperature=0
)

# response = llm.invoke("What is Neo4j?")
response = llm.invoke(template.format(fruit="apple"))

print(response)

