import os
import pickle
import sys


class DatasetWriter:

    def __init__(self, output_dir: str, file_name: str, max_buffer_size: int):
        self.output_dir = output_dir
        self.file_name = file_name
        self.seq = 0
        self.fp = None
        self.max_buffer_size = max_buffer_size
        self._next_file()
        self.buffer = []

    def _next_file(self):
        if self.fp is not None:
            self.fp.close()

        self.seq += 1
        self.fp = open(os.path.join(self.output_dir, f"{self.file_name}-{self.seq}.pkl"), "wb")

    def append(self, output: dict):
        if sys.getsizeof(self.buffer) > self.max_buffer_size:
            pickle.dump(self.buffer, self.fp)
            self._next_file()

        self.buffer.append(output)
