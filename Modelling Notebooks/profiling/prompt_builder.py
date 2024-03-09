from typing import Callable

from datasets import load_dataset
import os
from logging import info
from langchain.prompts import PromptTemplate

from transformers import PreTrainedTokenizer


class PromptBuilder:
    """
    build_fn should add two columns called 'prompt' and 'max_output_length'
    """

    def __init__(self, dataset_name: str, prompt_template: PromptTemplate, dataset_version: str, build_fn: Callable,
                 tokenizer: PreTrainedTokenizer, output_path: str, id_column: str):
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.build_fn = build_fn
        self.tokenizer = tokenizer
        self.output_path = output_path
        self.prompt_template = prompt_template
        self.id_column = id_column
        self.debug = os.environ.get('DEBUG', default=False)

    def tokenize(self, row):
        tokenized_input = self.tokenizer(row["prompt"], return_tensors="pt", padding=True)
        return {"tokens": tokenized_input, "input_length": len(tokenized_input[0])}

    def run(self):
        dataset = load_dataset(self.dataset_name, self.dataset_version)

        if self.debug:
            first_row = dataset["test"][0]
            info(f"Dataset row: {first_row}")

        info("Converting dataset to prompts")
        augmented_dataset = dataset.map(self.build_fn)
        augmented_dataset = augmented_dataset[[self.id_column, "prompt", "max_output_length"]]

        info("Converting prompts to tokens")
        encoded_dataset = augmented_dataset.map(lambda x: self.tokenize(x), batched=True)

        info("Saving tokenized dataset to disk")
        encoded_dataset.save_to_disk(self.output_path)
