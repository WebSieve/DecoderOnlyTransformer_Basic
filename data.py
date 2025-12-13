from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_seq_len: int,
        stride: Optional[int] = None,
        return_attention_mask: bool = True,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.stride = stride or max_seq_len
        self.max_seq_len = max_seq_len
        self.return_attention_mask = return_attention_mask

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.samples = []
        self._process_texts(texts=texts)

    def _process_texts(self, texts):
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) < 2:
                continue

            for i in range(0, len(tokens) - 1, self.stride):
                input_seq = tokens[i : i + self.max_seq_len]
                target_seq = tokens[i + 1 : i + self.max_seq_len + 1]

                if len(input_seq) < 2 or len(target_seq) < 2:
                    break

                input_len = len(input_seq)
                target_len = len(target_seq)

                if input_len < self.max_seq_len:
                    padding_len = self.max_seq_len - input_len
                    input_seq = input_seq + [self.tokenizer.pad_token_id] * padding_len

                if target_len < self.max_seq_len:
                    padding_len = self.max_seq_len - target_len
                    target_seq = (
                        target_seq + [self.tokenizer.pad_token_id] * padding_len
                    )

                input_seq = input_seq[: self.max_seq_len]
                target_seq = target_seq[: self.max_seq_len]

                self.samples.append(
                    {
                        "input_ids": input_seq,
                        "target_ids": target_seq,
                        "length": min(input_len, self.max_seq_len),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        target_ids = torch.tensor(item["target_ids"], dtype=torch.long)

        if self.return_attention_mask:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            return input_ids, target_ids, attention_mask

        return input_ids, target_ids


def Create_DataLoader(
    train_texts: List[str],
    val_texts: List[str],
    batch_size: int = 32,
    max_seq_len: int = 128,
    stride: Optional[int] = None,
    num_workers: int = 0,
    tokenizer_name: str = "gpt2",
) -> Tuple[DataLoader, DataLoader, object]:
    """
    Returns:
        train_loader, val_loader, tokenizer
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride,
        return_attention_mask=False,
    )

    val_dataset = TextDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=max_seq_len,
        return_attention_mask=False,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader, tokenizer
