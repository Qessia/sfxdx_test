# RAG LLM
This is a test task to SFXDX
Simple RAG system that can help you answer on questions related to PDF files

Made with Langchain and its' modules:
- FAISS vectorstore
- HuggingFaceEmbeddings
- LLM from GPT4All model:

## Preparation
- Download LLM model from GPT4All to `./model` directory: https://gpt4all.io/models/gguf/gpt4all-falcon-newbpe-q4_0.gguf
- Install requirements: `pip install -r requirements.txt`

## Usage
`python run.py {link} {question} {mode}`
where:
    - link - URL of PDF
    - question - your question "in brackets"
    - mode - "cpu" or "gpu" for embeddings