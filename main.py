import argparse
import sys
import io
import requests
from pypdf import PdfReader

headers = {'User-Agent': 'Mozilla/5.0 (X11; Windows; Windows x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'}


class AnswerSystem:
    def __init__(self, link, quest, mode):
        self.link = link
        self.quest = quest
        self.mode = mode
    
    def answer(self):
        print(f'Well, at least I got args: \nLink: {self.link}\nQuestion: {self.quest}\nMode: {self.mode}')
        self.load_pdf()
    
    def load_pdf(self):
        response = requests.get(url=self.link, headers=headers, timeout=120)
        on_fly_mem_obj = io.BytesIO(response.content)

        reader = PdfReader(on_fly_mem_obj)
        number_of_pages = len(reader.pages)
        page = reader.pages[0]
        text = page.extract_text()
        with open('test.txt', 'w', encoding='utf-8') as f:
            f.write(text)


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
