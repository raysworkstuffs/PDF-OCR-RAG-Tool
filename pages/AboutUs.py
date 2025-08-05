import streamlit as st

st.title("About Us")

st.write("""

My initial project scope on Padlet was a simple RAG that reads the FAQ section of Archives Online, and provides information to users of the RAG Tool. But later on I wanted to try something else that is way beyond my capability, but if successful, would be a practical tool that I will see myself using. That is to convert the PDFs of speeches on the Archives Online website into text using OCR function. Here's a link to the speeches where you can download the PDFs to try: https://www.nas.gov.sg/archivesonline/speeches/search-result?search-type=advanced&speaker=Ong+Teng+Cheong

I had some limitations - AISAY, which I had initially wanted to try out, were still in the midst of preparing the tool for use. Hence, I tried PyTesseract OCR as recommended by Mr Aldrian. Unfortunately, I soon found out that I had a bigger problem - the version of my MAC OS is too outdated. I am unable to download and install the Tesseract packages successfully in my local machine, and I am also unable to install and run Visual Studio Code. The browser version of VS Code is unable to use the terminal function, so that also takes away access. Hence I am trying a workaround by using Google Colab and running Streamlit from Colab instead. But this will mean that the Streamlit link can only be used when the Colab notebook is being run.

The main objective of this tool is to extract text from the PDF documents on Archives Online using OCR.

These are the features that I wanted:
- To allow users to upload their own PDFs into the tool
- To allow for uploading of multiple files
- To summarize the text extracted from the PDFs

Bonus:
- To display the full text of the document after OCR.
- Add a RAG (Retrieval-Augmented Generation) function after the summary is provided. This function should allow for multiple queries of all uploaded documents until the user types 'exit'.
- Include instructions for users on how to use the RAG functionality.

I MUST admit that Gemini helped to make the functions possible. At my current level, I am definitely unable to code all these by myself. As you can see from this notebook, Gemini was the one who helped me with much of the code.

""")
