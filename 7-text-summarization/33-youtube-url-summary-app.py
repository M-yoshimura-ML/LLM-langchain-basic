# pip install youtube_transcript_api validators pytube unstructured
import os

import validators
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


load_dotenv()

# streamlit app
st.set_page_config(page_title="LangChain: Summarize Text from YouTube or Website", page_icon="")
st.title("LangChain: Summarize Text From YouTube or Website")
st.subheader("Summarize URL")

# Get the Groq API key and URL(YouTube or Website)
with st.sidebar:
    groq_api_key = st.text_input("Groq API key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

llm = ChatGroq(model_name="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 1000 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])


if st.button("Summarize the Content from YouTube or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be YouTube video URL or Website URL.")

    else:
        try:
            with st.spinner("Waiting ..."):
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                   headers={"User-Agent": os.environ.get('USER_AGENT')})
                docs = loader.load()
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.error(f"There is something wrong: {e}")
