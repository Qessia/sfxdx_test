import argparse
import sys
import requests
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

headers = {'User-Agent': 'Mozilla/5.0 (X11; Windows; Windows x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'}


class AnswerSystem:
    """
    Retriever system class for answering questions in PDF text context
    """
    def __init__(self, link: str, quest: str, mode: str):
        """Initial method

        Args:
            link (str): URL of PDF you want to get context from
            quest (str): your question
            mode (str): Embeddings mode - whether to launch them on CPU or GPU
        """

        self.link = link
        self.quest = quest
        self.mode = 'cuda' if mode == 'gpu' else mode # torch uses word 'cuda' instead of 'gpu'

        # Uploading LLM from GPT4All
        self.llm = GPT4All(
            model="model/gpt4all-falcon-newbpe-q4_0.gguf",
            max_tokens=2048,
        )
    
    def answer(self):
        """method to answer the question"""

        self._load_pdf_to_FAISS()
        if self.__question_is_valid(): # validate if question is related to PDF context
            reply = self._invoke_chain()
        else:
            reply = 'No data to answer the question'
        print(reply)

    def _load_pdf_to_FAISS(self):
        """Internal method to load context from PDF to FAISS database using Embeddings"""

        # PDF parsing
        response = requests.get(url=self.link, headers=headers, timeout=120)

        # PyPDFLoader doesn't work with URLs, so we save it
        with open('temp.pdf', 'wb') as f:
            f.write(response.content)

        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()

        # splitting text of documents to chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # n_symbols
            chunk_overlap=20 # n_symbols shared by chunks
        )
        pages = text_splitter.split_documents(pages)

        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

        # Loading embeddings with specified device
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs= {'device': self.mode},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.db = FAISS.from_documents(documents=pages, embedding=embeddings)

    def __question_is_valid(self) -> bool:
        """Internal method to determine whether question is close enoung to PDF context

        Returns:
            bool: True if question is valid, else False
        """

        threshold = 1.6 # experimentally determined value
        docs_with_scores = self.db.similarity_search_with_score(self.quest) # returns list of tuple[Document, float]
        return docs_with_scores[0][1] < threshold # [0] index - doc with best (lowest) score, [1] index - score

    def _invoke_chain(self):
        """
        Internal method to invoke chain with retrieval prompt
        """

        # Switch FAISS db to retriever mode
        self.db = self.db.as_retriever(
            search_type="mmr", # Maximum Marginal Retrieval score
            search_kwargs={"k": 1} # Num of docs returned
        )

        # Q&A prompt
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": self.db, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return f'ANSWER:\n{chain.invoke(self.quest)}'


def main(link, quest, mode):
    system = AnswerSystem(link=link, quest=quest, mode=mode)
    system.answer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('link', type=str, help='URL of PDF file')
    parser.add_argument('question', type=str, help='Question you want to ask "in brackets"')
    parser.add_argument('mode', type=str, help='cpu or gpu')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    main(link=args.link, quest=args.question, mode=args.mode)
