import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.openai import OpenAI
# from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.callbacks import get_openai_callback

import pdf2image
import pytesseract



# Sidebar Contents
with st.sidebar:
    st.title('🤗💬 LLM Personal Chat App based on your PDF Docs!')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io)
    - [LangChain](https://www.langchain.com)
    - [FAISS Vectorestore](https://python.langchain.com/v0.1/docs/integrations/vectorstores/faiss/)
    - [Alternatively Pinecone Vectorestore](https://www.pinecone.io)
    - [OpanAI LLm Model](https://platform.openai.com/docs/models)
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Aiden')

def get_data_from_scanned_pdf(pdf_file_path):
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

    # Upload a pdf file
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
                extracted_text = get_data_from_scanned_pdf(filename)
                break
            else:
                extracted_text += page_content

        text_splitter  = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(extracted_text)

        #embeddings model
        embeddings = OpenAIEmbeddings()
        
        store_name = filename[:-4]
        store_path = "./docs/"
        if os.path.exists(f"{store_path}{store_name}.pkl"):
            Vectorestore = FAISS.load_local(folder_path=store_path, index_name=store_name, embeddings=embeddings, allow_dangerous_deserialization=True)
        else:
            #embeddings
            Vectorestore = FAISS.from_texts(chunks, embedding=embeddings)
            Vectorestore.save_local(folder_path=store_path, index_name=store_name)

        #Accept questions/query
        query = st.text_input("As your questuion about PDF file!")

        if query:
            # retriever = Vectorestore.as_retriever()
            docs = Vectorestore.similarity_search(query=query, k=3)
            
            llm = OpenAI(temperature=0)

            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()



