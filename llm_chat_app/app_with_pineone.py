import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pytesseract
import pdf2image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
import time

# Sidebar Contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App!')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - Streamlit
    - LangChain
    - OpenAI LLM Model
    ''')
    st.write('Made with ❤️ by Prompt Engineer')

def get_text_from_scanned_pdf(pdf_file_path):
    images = pdf2image.convert_from_path("./docs/" + pdf_file_path)
    extracted_text = ''
    
    for image in images:
        text = pytesseract.image_to_string(image=image, lang='eng')
        extracted_text += text
    
    processed_text = ' '.join(extracted_text.split())
    return processed_text

def main():
    st.header("Chat with PDF 💬")

    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_host = os.getenv("PINECONE_HOST")
    embeddings = OpenAIEmbeddings()

    pdf = st.file_uploader("Upload your PDF here!", type="pdf")

    if pdf is not None:
        st.success('File successfully uploaded', icon="✅")
        pdf_reader = PdfReader(pdf)
        filename = pdf.name
        num_pages = len(pdf_reader.pages)
        extracted_text = ''

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_content = page.extract_text()

            if not page_content:
                extracted_text = get_text_from_scanned_pdf(filename)
                break
            else:
                extracted_text += page_content

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(extracted_text)


        pc = Pinecone(api_key=pinecone_api_key)
        store_name = filename[:-4]

        if store_name in pc.list_indexes().names():
            st.write(f"{store_name} already exists!")
            Vectorestore = PineconeVectorStore.get_pinecone_index(store_name)
            query = "What is the capital of US?"
            embeded_query = embeddings.embed_query(text=query)
            response = Vectorestore.query(top_k=1, vector=embeded_query)
        #     docsearch = PineconeVectorStore.from_existing_index(index_name=store_name, embedding=embeddings)
        #     docs = docsearch.similarity_search(query=query)
            st.write(response)
        else:
            st.write(f"{store_name} index does not exist, creating a new index as {store_name}")
            # PineconeVectorStore.get_pinecone_index
            pc.create_index(
                name=store_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            while not pc.describe_index(store_name).status['ready']:
                time.sleep(1)

            Vectorestore = PineconeVectorStore.from_texts(
                chunks,
                index_name=store_name,
                embedding=embeddings,
            )
            while not pc.describe_index(store_name).status['ready']:
                time.sleep(1)
            query = "What is the capital of US?"
            response = Vectorestore.similarity_search(query=query)
            st.write(response)



if __name__ == '__main__':
    main()
