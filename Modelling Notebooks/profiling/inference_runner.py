import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from dataset_writer import DatasetWriter


class InferenceRunner:
    def __init__(self, encoded_dataset_path: str, model: PreTrainedModel, batch_size: int, device: torch.device,
                 output_dir: str, output_filename: str, max_file_size: int):
        self.dataset = Dataset.load_from_disk(encoded_dataset_path)
        self.batch_size = batch_size
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size)
        self.model = model.to(device)
        self.dataset_writer = DatasetWriter(output_dir, output_filename, max_file_size)

    @torch.no_grad()
    def run(self):
        self.model.eval()
        for batch in self.data_loader:
            tokens = batch["tokens"].to(self.model.device)
            lengths = batch["input_lengths"]
            ids_batch = batch["id"]

            output = self.model.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'],
                                         return_dict_in_generate=True, output_router_logits=True)
            self.dataset_writer.append(output)












