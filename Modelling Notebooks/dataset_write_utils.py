# Source: https://stackoverflow.com/questions/49058313/how-to-write-to-multiple-csv-files-with-a-max-size
from typing import IO
from io import TextIOBase

class SizeCappingFileWriter(TextIOBase, IO[str]):
    def __init__(self, filename, maxsize):
        self.filename = filename
        self.maxsize = maxsize
        self.count = 0
        self.file = None

    def write(self, value):
        if not self.file:
            self.count += 1
            self.file = open(
                f'{self.filename}_{self.count}.csv', 'w', encoding='utf-8', newline='')
        self.file.write(value)
        if value.endswith('\n') and self.file.tell() > self.maxsize:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()