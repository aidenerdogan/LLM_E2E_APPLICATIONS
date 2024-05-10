**LLM Chat App (RAG on your PDF docs)**
=====================================================

**Overview**
-----------

The LLM Chat App is a Streamlit application that integrates LangChain and FAISS to create a language model-based chatbot. This chatbot is designed to interact with users based on the content of PDF files uploaded by the user. The application uses OpenAI's embeddings model to generate embeddings from the extracted text and stores them in a vector store using FAISS. The chatbot then uses these embeddings to retrieve relevant documents and generate responses to user queries.

**Features**
------------

- **PDF File Upload**: Users can upload PDF files, which are then processed to extract text content.
- **Text Processing**: The extracted text is processed using a recursive character text splitter to split the text into manageable chunks.
- **Embeddings Generation**: The text chunks are used to generate embeddings using OpenAI's embeddings model.
- **Vector Store**: The embeddings are stored in a vector store using FAISS, which is a high-performance vector store for similarity search.
- **Query Processing**: The application accepts user queries or questions about the PDF file content.
- **Retrieval and Response Generation**: The vector store is used to retrieve relevant documents based on the user query. The retrieved documents are then used to generate a response using an OpenAI language model.
- **Chat Interface**: The application provides a simple chat interface where users can interact with the chatbot.

**Getting Started**
-------------------

### Prerequisites

- Python 3.8 or higher
- Streamlit
- LangChain
- FAISS
- OpenAI

### Installation

1. Clone the repository: `git clone https://github.com/aidenerdogan/LLM_E2E_APPLICATIONS.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

### Usage

1. Upload a PDF file using the chat interface.
2. Enter a query or question about the PDF file content.
3. The chatbot will generate a response based on the uploaded PDF file and the user query.

**Contributing**
--------------

Contributions are welcome. If you'd like to contribute to the project, please fork the repository and create a pull request with your changes.

**License**
---------

The LLM Chat App is licensed under the MIT License.

**Acknowledgments**
----------------

This project was inspired by the work of the LangChain and FAISS communities. Special thanks to OpenAI for providing the embeddings model used in this application.

**Contact**
---------

If you have any questions or would like to report an issue.