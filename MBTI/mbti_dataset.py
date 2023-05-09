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
        self.age    = df['Age']
        self.gender = df['Gender']
        self.q_num  = df['Q_number']
        self.label  = df[target] if target is not None else None

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        sample = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        sample['age']     = torch.tensor(self.age[idx])
        sample['gender']  = torch.tensor(self.gender[idx])
        sample['q_num']   = torch.tensor(self.q_num[idx])
        if self.label is not None:
            sample["label"] = torch.tensor(self.label[idx])

        return sample