# Document Q&A with RAG

## Project Overview

This project is a Retrieval Augmented Generation (RAG) application built with Streamlit, LangChain, FAISS, and OpenAI’s Chat API. The application allows users to upload multiple documents (supporting `.txt`, `.md`, and `.pdf` files), automatically extracts and processes the text content by chunking large files into manageable pieces, and then enables a conversational interface where users can ask questions about the uploaded documents. The system retrieves the most relevant text chunks using a FAISS vector store and generates contextually relevant responses with OpenAI’s Chat API.

## Features

- **Multi-File Support:** Upload multiple documents simultaneously.
- **File Format Support:** Handles `.txt`, `.md`, and `.pdf` files.
- **Document Chunking:** Automatically splits large documents for better retrieval.
- **Retrieval Mechanism:** Uses FAISS for efficient similarity search among document chunks.
- **Conversational Chat Interface:** Allows users to interact with document content through a chat-based UI.
- **Error Handling:** Robust error logging and user feedback during file processing and response generation.

## Prerequisites

Before running the project, ensure you have:

- [Docker](https://www.docker.com/get-started) (with Docker Desktop running)
- [Visual Studio Code](https://code.visualstudio.com/) with the Remote - Containers extension (if using a devcontainer)
- [Git](https://git-scm.com/)
- [Python 3.8+](https://www.python.org/downloads/)
- An OpenAI API Key

###  Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/yy2358/info5940ASMT1.git
cd info5940ASMT1
   ```
### How to Use the Application

## Upload Documents

Use the file uploader to select multiple documents (.txt, .md, or .pdf).
The system extracts text from each file, splits it into smaller chunks, and updates the FAISS vector store.
Ask Questions

After the documents are uploaded, type a question in the chat input field.
The system retrieves the most relevant chunks from the vector store, then uses OpenAI’s Chat API to generate a response.
View Chat History

The conversation is displayed above the input box using Streamlit’s chat features.

### Extra Info
  - Use starter template with Docker and .devcontainer configurations to streamline the development environment.
  - Modified the "chat_with-pdf.py" file and changed to assignmentmain from the lecture 5 branch.
  - Excluded most files from lecture 5 branch to avoid accident leak


Happy coding! 🚀