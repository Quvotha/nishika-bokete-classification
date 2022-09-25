from typing import Literal, Sequence, Tuple

from transformers import BertModel, GPT2Model, RobertaModel
from transformers import AutoTokenizer, T5Tokenizer
import torch

SequenceModelName = Literal[
    "rinna/japanese-gpt2-medium",
    "rinna/japanese-roberta-base",
    "cl-tohoku/bert-base-japanese-v2",
]


def get_model_and_tokenizer(model_name: SequenceModelName) -> Tuple[object, object]:
    """訓練済みの文章分類モデルと、モデルに対応した Tokenizer を得る。

    Parameters
    ----------
    model_name : SequenceModelName
        モデル名称。

    Returns
    -------
    model, tokenizer: Tuple[object, object]
        訓練済みの文章分類モデルと、モデルに対応した Tokenizer.

    Raises
    ------
    ValueError
        `model_name` に不適切なモデル名を指定した場合。
    """
    if model_name == "rinna/japanese-gpt2-medium":
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        tokenizer.do_lower_case = True
        model = GPT2Model.from_pretrained(model_name)
    elif model_name == "rinna/japanese-roberta-base":
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        tokenizer.do_lower_case = True
        model = RobertaModel.from_pretrained(model_name)
    elif model_name == "cl-tohoku/bert-base-japanese-v2":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    else:
        raise ValueError(f'"{model_name}" is invalid')
    return model, tokenizer


class SequenceVectorizer(torch.nn.Module):
    # https://www.guruguru.science/competitions/16/discussions/fb792c87-6bad-445d-aa34-b4118fc378c1/

    def __init__(self, model_name: SequenceModelName):
        super(SequenceVectorizer, self).__init__()
        model, tokenizer = get_model_and_tokenizer(model_name)
        self.tokenizer = tokenizer
        self.backbone = model
        self.model_name = model_name

    def forward(self, sequence: str) -> torch.Tensor:
        tokenized_sequence = self.tokenizer(sequence, return_tensors="pt", padding=True)
        outputs = self.backbone(**tokenized_sequence)
        return outputs["last_hidden_state"][:, 0, :]

    @property
    def ndim(self) -> int:
        if self.model_name == "rinna/japanese-gpt2-medium":
            return 1024
        elif self.model_name in (
            "rinna/japanese-roberta-base",
            "cl-tohoku/bert-base-japanese-v2",
        ):
            return 768


class SequenceClassifier(torch.nn.Module):
    def __init__(self, model_name: SequenceModelName, n_classes: int = 2):
        super(SequenceClassifier, self).__init__()
        self.vectorizer = SequenceVectorizer(model_name)
        self.activation = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(
            self.vectorizer.ndim, 1 if n_classes == 2 else n_classes
        )
        self.model_name = model_name
        self.n_classes = n_classes

    def forward(self, sequences: Sequence[str]) -> torch.Tensor:
        vector = self.vectorizer(sequences)
        return self.classifier(self.activation(vector))


if __name__ == "__main__":
    texts = ["Nishika コンペティション", "Noshika bokete", "Nishika Sさん"]
    for m in [
        "rinna/japanese-gpt2-medium",
        "rinna/japanese-roberta-base",
        "cl-tohoku/bert-base-japanese-v2",
    ]:
        vectorizer = SequenceVectorizer(m)
        vector = vectorizer(texts)
        assert vector.shape[-1] == vectorizer.ndim, (vector.shape, vectorizer.ndim)

        classifier = SequenceClassifier(m)
        prediction = classifier(texts)
