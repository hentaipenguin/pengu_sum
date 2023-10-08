import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import enchant
import nltk
import wordninja
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline

from converter import PDFToTextConverter


class PDFSummarizer(PDFToTextConverter):
    """
    PDFSummarizer is a class that summarizes PDF documents. It uses the NLTK and Transformers libraries to tokenize and filter sentences and words in the text, and to generate a summary of the document.

    The class inherits from PDFToTextConverter, which is a class that converts PDF files to text. The summarized text can be exported to a file using the export() method.

    Attributes:
        CHUNK_SIZE (int): The size of the text chunks to be processed concurrently. Default is 4096.
        MAX_LENGTH (int): The maximum length of the generated summary. Default is 100.
        MIN_LENGTH (int): The minimum length of the generated summary. Default is 30.
        NUM_SENTENCES (int): The number of sentences to include in the summary. Default is 50.
    """
    CHUNK_SIZE = 4096
    MAX_LENGTH = 80  # modify
    MIN_LENGTH = 30
    NUM_SENTENCES = 20

    def __init__(self, filename: str) -> None:
        super().__init__(filename)
        self._download()
        self.stop_words = set(stopwords.words("english"))
        self.text = self._sanitize(self.text)
        self.chunks = PDFSummarizer.split_text(self.text, self.CHUNK_SIZE)
        self.summary = ""
        self.summarizer = pipeline(task="summarization",
                                   model="sshleifer/distilbart-cnn-12-6")

    def _sanitize(self, text):
        return re.sub(r"\[\d+\]", "",
                      re.sub(r"http\S+", "", text, flags=re.MULTILINE))

    def _download(self) -> None:
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("maxent_ne_chunker")
        nltk.download("words")

    @staticmethod
    def split_text(text: str, chunk_size: int) -> list:
        return [
            text[i:i + chunk_size] for i in range(0, len(text), chunk_size)
        ]

    def process_concurrently(self, tokens: list, num_threads: int,
                             operation) -> None:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                offset = i * self.CHUNK_SIZE
                chunk = list(islice(tokens, offset, offset + self.CHUNK_SIZE))
                futures.append(executor.submit(operation, chunk))
            return [
                sentence for future in futures for sentence in future.result()
            ]

    @staticmethod
    def tokenize_sentences(chunks: list) -> list:
        return [token for chunk in chunks for token in sent_tokenize(chunk)]

    @staticmethod
    def tokenize_words(chunks: list) -> list:
        return [token for chunk in chunks for token in word_tokenize(chunk)]

    @staticmethod
    def filter_sentences(chunks: list) -> list:
        return [
            token for token in chunks if len(token) > 1 and not any(
                word in token for word in {"Figure", "Fig", "Tab", "Table"})
        ]

    def filter_words(self, chunks: list) -> list:
        return [
            word for word in chunks
            if word not in self.stop_words and len(word) > 1
        ]

    def summarize(self, quiet=False) -> None:
        self.sentences = self.process_concurrently(
            self.chunks,
            num_threads=4,
            operation=PDFSummarizer.tokenize_sentences)
        self.words = self.process_concurrently(
            self.sentences,
            num_threads=4,
            operation=PDFSummarizer.tokenize_words)

        self.filtered_sentences = self.process_concurrently(
            self.sentences,
            num_threads=4,
            operation=PDFSummarizer.filter_sentences)
        self.filtered_words = self.process_concurrently(
            self.words, num_threads=4, operation=self.filter_words)

        if not quiet:
            print("===================================")
            print(f"Sentences length: {len(self.sentences)}")
            print(f"Words length: {len(self.words)}")
            print(f"Filtered sentences length: {len(self.filtered_sentences)}")
            print(f"Filtered words length: {len(self.filtered_words)}")
            print("===================================")

        frequency_table = FreqDist(self.filtered_words)

        scores = OrderedDict()
        for sentence in self.filtered_sentences:
            sentence_words = word_tokenize(sentence.lower())
            sentence_score = sum([
                frequency_table.freq(word) for word in sentence_words
                if word in self.filtered_words
            ])
            scores[sentence] = sentence_score

        self.CHUNK_SIZE = 1024

        raw_summary_chunks = PDFSummarizer.split_text(
            "".join([
                sentence for sentence in scores.keys() if sentence in sorted(
                    scores, key=scores.get, reverse=True)[:self.NUM_SENTENCES]
            ]), self.CHUNK_SIZE)

        with ThreadPoolExecutor() as executor:
            futures = list(
                executor.map(self._summarize_chunk, raw_summary_chunks))
            self.summary = "".join(futures)
            self._correct_summary()

    def _summarize_chunk(self, chunk) -> str:
        return self.summarizer(chunk,
                               max_length=self.MAX_LENGTH,
                               min_length=self.MIN_LENGTH,
                               do_sample=True)[0]["summary_text"]

    def _correct_summary(self) -> None:
        eng_dict = enchant.Dict("en_US")
        words = word_tokenize(self.summary)
        clean_summary = []
        for word in words:
            split_words = wordninja.split(word)
            if len(word) > 5 and not eng_dict.check(word) and all(
                    eng_dict.check(w) for w in split_words):
                clean_summary.extend(split_words)
            else:
                clean_summary.append(word)
        self.summary = re.sub("\s+", " ", " ".join(clean_summary)).strip()

    def export(self, filename: str) -> None:
        with open(filename, mode="w", encoding="utf-8") as f:
            f.write(self.summary)

    def suma(self):
        return self.summary
