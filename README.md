# Research Question Answering App

This is a Streamlit application that allows users to upload a PDF research, input a question about the research, and receive a detailed summary or answer based on the content of the PDF.

## Features

- **Upload PDF**: Upload a PDF file to be processed.
- **Ask Questions**: Enter questions about the content of the PDF.
- **Get Answers**: Receive detailed answers or summaries based on the PDF content.

## Requirements

- Python 3.7 or higher
- Required Python libraries:
  - `streamlit`
  - `pdfplumber`
  - `langchain_community`
  - `langchain`
  - `langchain_chroma`
  - `langchain_google_genai`
  - `langchain_core`
  - `python-dotenv`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shimaa83/LLMChat.git
   cd LLMChat
2. Create a .env file in the root directory of the project and add your Google API key:
      - GOOGLE_API_KEY=your_google_api_key_here
3. Start the Streamlit app:
  -  streamlit run chat.py






