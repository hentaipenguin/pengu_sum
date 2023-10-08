import os
import pypdf
import re


class PDFToTextConverter:
    """
    A class that converts .pdf files to .txt files.

    Attributes:
        filename (str): The path to the .pdf file.
        text (str): The content of the .pdf file.
    """
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    def __init__(self, filename: str) -> None:
        self.filename = self._validate_file(filename)
        self.text = self._read_file(filename)

    def _validate_file(self, filename: str) -> str:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file '{filename}' does not exist.")
        with open(filename, mode="rb") as f:
            if not f.read(4) == b"%PDF":
                raise ValueError(f"The file '{filename}' is not a PDF file.")
        return filename

    def _read_file(self, filename: str) -> str:
        self._validate_file(filename)
        with open(filename, mode="rb") as f:
            reader = pypdf.PdfReader(f)
            writer = pypdf.PdfWriter(clone_from=reader)
            writer.remove_annotations(subtypes=None)
        return self._remove_noise(" ".join(page.extract_text()
                                  for page in writer.pages))

    def _remove_noise(self, text):
        index = text.lower().rfind("references") or text.lower().rfind(
            "bibliography")
        if (index != -1):
            text = text[:index]
        text = re.sub(self.url_pattern, "", re.sub(self.email_pattern, "",
                                                   text))
        return text.replace("-", "")

    def export(self, filename: str) -> None:
        with open(filename, mode="w", encoding="utf-8") as f:
            f.write(self.text)

    def disp(self):
        return self.text