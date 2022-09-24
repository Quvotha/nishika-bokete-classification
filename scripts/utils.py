import os
import random

import numpy as np
import pandas as pd
import torch

from scripts.images import read_jpg


class NishikaBoketeDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str):
        self.ids = df["id"].tolist()
        self.filenames = df["odai_photo_file_name"].tolist()
        self.texts = df["text"].tolist()
        self.labels = df["is_laugh"].tolist() if "is_laugh" in df else None
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        id_ = self.ids[i]
        image_filepath = os.path.join(self.image_dir, self.filenames[i])
        image = read_jpg(image_filepath)
        text = self.texts[i]
        label = None if self.labels is None else self.labels[i]
        return id_, image, text, label


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
