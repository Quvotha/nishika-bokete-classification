import os
import random
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch

from scripts.images import read_jpg

# id, image tensor, text, label if training set otherwise None
NishikaBoketeData = Tuple[str, torch.Tensor, str, Union[int, None]]

# list of id, list of image tensor, list of text, list of label if training set otherwise None
NishikaBoketeBatch = Tuple[
    List[str], List[torch.Tensor], List[str], Union[List[int], None]
]


class NishikaBoketeDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str):
        self.ids = df["id"].tolist()
        self.filenames = df["odai_photo_file_name"].tolist()
        self.texts = df["text"].tolist()
        self.labels = df["is_laugh"].tolist() if "is_laugh" in df else None
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> NishikaBoketeData:
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


def collate_fn(batch: List[NishikaBoketeData]) -> NishikaBoketeBatch:
    ids = []
    image_tensors = []
    texts = []
    labels = []
    for id_, image_tensor, text, label_or_None in batch:
        ids.append(id_)
        image_tensors.append(image_tensor)
        texts.append(text)
        if label_or_None is not None:
            labels.append(label_or_None)
    if labels:
        return ids, image_tensors, texts, labels
    else:
        return ids, image_tensors, texts, None
