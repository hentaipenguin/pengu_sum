from transformers import AutoTokenizer, BigBirdPegasusForConditionalGeneration

from converter import PDFToTextConverter


class PDFSummarizer(PDFToTextConverter):
    """
    A PDF summarizer that uses BigBirdPegasusForConditionalGeneration from Hugging Face's Transformers library to generate summaries.

    Args:
        filename (str): The name of the PDF file to summarize.
        model (str): The name or path of the pre-trained BigBirdPegasus model to use.

    Attributes:
        CHUNK_SIZE (int): The chunk size to use when processing the PDF file. Defaults to 4096.
        MAX_LENGTH (int): The maximum length of the summary. Defaults to 100.
        tokenizer: The tokenizer to use for encoding the text. Initialized in __init__().
        model: The pre-trained BigBirdPegasus model to use for generating the summaries. Initialized in __init__().
    """
    CHUNK_SIZE = 4096
    MAX_LENGTH = 100

    def __init__(self, filename: str, model="google/bigbird-pegasus-large-arxiv") -> None:
        super().__init__(filename)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(
            model)

    def _split_text(self) -> None:
        self.chunks = [
            self.text[i:i + self.CHUNK_SIZE]
            for i in range(0, len(self.text), self.CHUNK_SIZE)
        ]

    def summarize(self, quiet=False) -> None:
        self._split_text()
        self.summary = ""
        for i, chunk in enumerate(self.chunks):
            if not quiet:
                print(f"Processing chunk {i + 1}/{len(self.chunks)}...")
            inputs = self.tokenizer.encode(chunk,
                                           return_tensors="pt",
                                           max_length=self.CHUNK_SIZE,
                                           truncation=True)
            summary_ids = self.model.generate(inputs,
                                              num_beams=4,
                                              max_length=self.MAX_LENGTH,
                                              early_stopping=True)
            self.summary += self.tokenizer.decode(summary_ids[0],
                                                  skip_special_tokens=True)

    def export(self, filename: str) -> None:
        with open(filename, mode="w", encoding="utf-8") as f:
            f.write(self.summary)

    def suma(self):
        return self.summary
