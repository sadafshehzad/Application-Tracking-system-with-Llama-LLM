import streamlit as st
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


load_dotenv() ## load all our environment variables

llm = ChatGroq(temperature=0, api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")


def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

#Prompt Template

prompt_extract = PromptTemplate.from_template(

    """
    Hey Act Like a skilled or very experience ATS(Application Tracking System)
    with a deep understanding of tech field,software engineering,data science ,data analyst
    and big data engineer. Your task is to evaluate the resume based on the given job description.
    You must consider the job market is very competitive and you should provide 
    best assistance for improving the resumes. Assign the percentage Matching based 
    on Jd and
    the missing keywords with high accuracy
    resume:{text}
    description:{jd}

    I want the response in one single string having the structure
    {{"JD Match":"%","MissingKeywords:[]","Profile Summary":""}}
    """
)

## streamlit app
st.title("Smart ATS")
st.text("Improve Your Resume ATS")
jd=st.text_area("Paste the Job Description")
uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please uplaod the pdf")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        text=input_pdf_text(uploaded_file)
        chain_extract = prompt_extract | llm 
        response = chain_extract.invoke({"jd":jd,"text": text})
        res=response.content
        st.subheader(res)