import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.environ.get('GROQ_API_KEY'))

messages = [
    SystemMessage(content="Translate the following from English to French."),
    HumanMessage(content="Hello How are you ?")
]

result = model.invoke(messages)

print(result)

output_parser = StrOutputParser()

str_result = output_parser.invoke(result)
# print(str_result)

# Using LCEL - chain the components
chain = model | output_parser
chain_result = chain.invoke(messages)
print(chain_result)

# Prompt Template
generic_template = "Translate the following into {language}"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
    ]
)

print("\n==============================\n")

prompt_result = prompt.invoke({"language": "French", "text": "Hello"})
print(prompt_result)

print("\n==============================\n")

chain = prompt | model | output_parser
chain_result = chain.invoke({"language": "French", "text": "Hello"})
print(chain_result)
