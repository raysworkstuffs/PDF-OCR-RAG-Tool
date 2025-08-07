import os
import openai
import pdf2image
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import streamlit as st
import tiktoken

def pdf_to_img(pdf_file):
    """Converts a PDF file to a list of PIL Images."""
    return pdf2image.convert_from_path(pdf_file, poppler_path='/usr/bin')


def ocr_core(file):
    """Performs OCR on a single image and returns the extracted text using Tesseract or OCR API."""
    # Try Tesseract first
    try:
        return pytesseract.image_to_string(file)
    except Exception as e:
        st.warning("Tesseract not available, using OCR API instead.")
        return ocr_space_image(file)


def ocr_space_image(image):
    """Performs OCR using the OCR.Space API."""
    api_key = st.secrets["OCR_API_KEY"]
    url = 'https://api.ocr.space/parse/image'
    # Convert PIL Image to bytes
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    files = {'file': buffered}
    data = {'isOverlayRequired': False, 'OCREngine': 2}
    headers = {'apikey': api_key}
    response = requests.post(url, files=files, data=data, headers=headers)
    result = response.json()
    try:
        return result['ParsedResults'][0]['ParsedText']
    except Exception:
        return ""


def extract_text_from_pdf(pdf_file):
    """Extracts text from all pages of a PDF file."""
    images = pdf_to_img(pdf_file)
    extracted_text = ""
    for img in images:
        extracted_text += ocr_core(img) + "\n\n"
    return extracted_text


def count_tokens(text: str) -> int:
    """Counts the number of tokens in a string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def main():
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

    st.title("PDF OCR & Summarize & RAG")

    with st.expander("IMPORTANT NOTICE"):
        st.write("""
        This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

        Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

        Always consult with qualified professionals for accurate and personalized advice.
        """)

    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if uploaded_files:
        all_pdf_text = ""
        for uploaded_file in uploaded_files:
            try:
                file_path = os.path.abspath(uploaded_file.name)
                with open(file_path, "wb") as f:
                     f.write(uploaded_file.getbuffer())
                print("Processing file:", file_path)
                pdf_text = extract_text_from_pdf(file_path)
                all_pdf_text += pdf_text
                st.success(f"Successfully extracted text from '{uploaded_file.name}'.")
            except Exception as e:
                st.error(f"An error occurred while processing '{uploaded_file.name}': {e}")

        if all_pdf_text:
            st.subheader("Extracted Text")
            st.text_area("Full text from PDF(s)", all_pdf_text, height=300)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_text(all_pdf_text)

            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectordb = Chroma.from_texts(splits, embeddings)

            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm, retriever=vectordb.as_retriever()
            )
            query = "Summarize the content of the PDF(s)."
            summary = st.session_state.qa_chain.run(query)
            st.write("---")
            st.subheader("Summary")
            st.write(summary)
            st.write("---")
            st.write(f"Token count: {count_tokens(all_pdf_text)}")

    if st.session_state.qa_chain:
        st.subheader("Chat with your Documents (RAG)")
        st.write("Ask questions about the uploaded documents. Type 'exit' to stop.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if prompt.lower() != "exit":
                with st.chat_message("assistant"):
                    response = st.session_state.qa_chain.run(prompt)
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        password = st.text_input("Password", type="password")
        if password:
            if password == st.secrets["APP_PASSWORD"]:
                st.session_state["password_correct"] = True
                st.success("Password correct. Please reload the page or interact with the app.")
            else:
                st.error("😕 Password incorrect")
        return False
    else:
        return True

if check_password():
    main()
