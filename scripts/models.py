from typing import Iterable, Tuple

import torch

from scripts.image_models import ImageClassifier, ImageModelName
from scripts.sequence_models import SequenceClassifier, SequenceModelName


class BoketeClassifier(torch.nn.Module):
    def __init__(
        self,
        image_model_name: ImageModelName,
        sequence_model_name: SequenceModelName,
        n_classes: int = 2,
    ):
        super(BoketeClassifier, self).__init__()
        # 画像のベクトル化モデルの読み込み
        image_classifier = ImageClassifier(image_model_name)
        self.image_vectorizer = image_classifier.vectorizer
        # テキストのベクトル化モデルの読み込み
        sequence_classifier = SequenceClassifier(sequence_model_name)
        self.sequence_vectorizer = sequence_classifier.vectorizer
        # 最終層
        ndim = self.image_vectorizer.ndim + self.sequence_vectorizer.ndim
        self.output = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(ndim, ndim),
            torch.nn.ReLU(),
            torch.nn.Linear(ndim, 1 if n_classes == 2 else n_classes),
        )
        # その他
        self.image_model_name = image_model_name
        self.sequence_model_name = sequence_model_name
        self.n_classes = n_classes

    def forward(self, image_tensor: torch.Tensor, tokenized) -> torch.Tensor:
        image_vector = self.image_vectorizer(image_tensor)
        sequence_vector = self.sequence_vectorizer(tokenized)
        assert image_vector.shape[0] == sequence_vector.shape[0]
        vector = torch.cat([image_vector, sequence_vector], dim=1)
        return self.output(vector)

    def preprocess(
        self, image_tensors: Iterable[torch.Tensor], texts: Iterable[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images_preprocessed = self.image_vectorizer.preprocess(image_tensors)
        tokenized = self.sequence_vectorizer.preprocess(texts)
        return images_preprocessed, tokenized

    @property
    def transforms(self):
        return self.image_vectorizer.transforms

    @property
    def tokenizer(self):
        return self.sequence_vectorizer.tokenizer


if __name__ == "__main__":

    texts = ["Nishikaボケて", "いろはにほへとちりぬるを", "仰げば尊し", "ゴンザレス井上"]
    image_tensors = torch.Tensor(size=(len(texts), 3, 224, 224))

    image_model_names = (
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "resnet18",
        "resnet34",
        "vgg16",
        "vgg16_bn",
    )
    sequence_model_names = (
        "rinna/japanese-gpt2-medium",
        "rinna/japanese-roberta-base",
        "cl-tohoku/bert-base-japanese-v2",
    )
    for image_model_name in image_model_names:
        for sequence_model_name in sequence_model_names:
            model = BoketeClassifier(image_model_name, sequence_model_name)
            images_transformed, tokenized = model.preprocess(image_tensors, texts)
            model(images_transformed, tokenized)
