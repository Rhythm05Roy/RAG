# Chat with Documents using Llama 3.1

This project demonstrates a conversational interface that enables users to interact with unstructured PDF documents using the Llama 3.1 large language model. It uses LangChain, FAISS, and Streamlit for building a robust and interactive chat application.

## Features
- **Document Loading**: Upload and process unstructured PDF documents.
- **Vector Store Setup**: Convert documents into vector embeddings using FAISS and HuggingFace embeddings.
- **Conversational Retrieval Chain**: Interact with documents using Llama 3.1 with a memory buffer for seamless conversations.
- **Streamlit Interface**: Simple and user-friendly UI for uploading files and querying documents.

## Requirements
- Python 3.9+
- Required Libraries:
  - `os`
  - `python-dotenv`
  - `streamlit`
  - `langchain_community`
  - `langchain_groq`
  - `faiss`
  - `huggingface`

## Installation
1. **Clone the Repository:**
   ```bash
   https://github.com/Rhythm05Roy/RAG.git
   cd RAG
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables:**
   - Create a `.env` file in the project directory.
   - Add necessary environment variables (e.g., API keys for HuggingFace or LangChain).

## How to Run
1. **Start the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

2. **Upload PDF Files:**
   - Use the file uploader in the app to upload unstructured PDF documents.

3. **Ask Questions:**
   - Type questions related to the document in the chat input box.
   - Get relevant and accurate responses using Llama 3.1.

## Code Overview
### Key Components

1. **Dependencies**
   - Importing required libraries such as `langchain_community`, `FAISS`, `HuggingFaceEmbeddings`, and `streamlit`.

2. **Document Loading**
   - Uses `PyPDFLoader` from `langchain_community` to load unstructured PDF documents.

3. **Vector Store Setup**
   - Splits documents into chunks using `CharacterTextSplitter`.
   - Converts chunks into vector embeddings using HuggingFace embeddings.
   - Stores vectors in FAISS for efficient retrieval.

4. **Conversational Retrieval Chain**
   - Creates a chain with Llama 3.1 for question answering.
   - Uses a `ConversationBufferMemory` for maintaining chat context.

5. **Streamlit Interface**
   - Allows users to upload documents and interact with the chatbot.
   - Maintains chat history for a smooth user experience.

## Directory Structure
```
├── app.py                 # Main application file
├── requirements.txt       # List of dependencies
├── .env                   # Environment variables
```

## Technologies Used
- **LangChain**: For building conversational chains and document processing.
- **FAISS**: Efficient vector search and retrieval.
- **HuggingFace**: Embedding generation.
- **Streamlit**: Interactive web application framework.
- **Llama 3.1**: Large language model for conversational AI.

## Future Improvements
- Support for multiple document formats (e.g., DOCX, TXT).
- Integration with more LLMs.
- Enhanced memory and context handling.

## Credits
- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)

## License
This project is licensed under the MIT License.

