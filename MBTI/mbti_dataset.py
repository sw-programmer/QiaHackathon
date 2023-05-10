import torch
from torch.utils.data import Dataset

# Define dataset
class MBTIDataset(Dataset):
    def __init__(
        self,
        encodings,
        df,
        target = None
        ):
        self.encodings = encodings
        self.df = df
        self.selected = ['Age', 'Gender', 'Q_number', '<그렇다>', '<중립>', '<아니다>']
        self.label  = df[target] if target is not None else None

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        sample = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        for col in self.selected:
            sample[col] = torch.tensor(self.df[col][idx])
        # Only for training
        if self.label is not None:              
            sample["label"] = torch.tensor(self.label[idx])

        return sample