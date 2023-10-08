from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from converter import PDFToTextConverter


class PDFSummarizer(PDFToTextConverter):
    """
    A class that can be used to summarize text extracted from a PDF file.
    Inherits from PDFToTextConverter, which extracts the text from the PDF file.

    Attributes:
    - LANGUAGE (str): The language to use for summarization. Default is "english".
    - NUM_SENTENCES (int): The number of sentences to include in the summary. Default is 20.
    """
    LANGUAGE = "english"
    NUM_SENTENCES = 20

    def __init__(self, filename) -> None:
        super().__init__(filename)
        self.summary = ""

    def summarize(self) -> None:
        stemmer = Stemmer(self.LANGUAGE)
        summarizer = Summarizer(stemmer)
        parser = PlaintextParser.from_string(self.text,
                                             Tokenizer(self.LANGUAGE))
        summarizer.stop_words = get_stop_words(self.LANGUAGE)
        for sentence in summarizer(parser.document, self.NUM_SENTENCES):
            self.summary += sentence._text

    def export(self, filename: str) -> None:
        with open(filename, mode="w", encoding="utf-8") as f:
            f.write(self.summary)

    def suma(self):
        return self.summary