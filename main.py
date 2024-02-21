import argparse
import sys
import requests
import os
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

headers = {'User-Agent': 'Mozilla/5.0 (X11; Windows; Windows x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'}


class AnswerSystem:
    def __init__(self, link, quest, mode):
        self.link = link
        self.quest = quest
        self.mode = mode

        self.llm = GPT4All(
            model="model/gpt4all-falcon-newbpe-q4_0.gguf",
            max_tokens=2048,
        )
    
    def answer(self):
        self.load_pdf_to_FAISS()
        self.make_chain()

    def load_pdf_to_FAISS(self):
        response = requests.get(url=self.link, headers=headers, timeout=120)
        with open('temp.pdf', 'wb') as f:
            f.write(response.content)

        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs= {'device': self.mode},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.db = FAISS.from_documents(documents=pages, embedding=embeddings)

    def make_chain(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        docs = self.db.similarity_search(self.quest)[:2]

        rag_prompt = hub.pull("rlm/rag-prompt")

        chain = (
            RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        res = chain.invoke({"context": docs, "question": self.quest})
        print(res)


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
