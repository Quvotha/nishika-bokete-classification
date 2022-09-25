from typing import Literal, Tuple

import torch
import torchvision.models


ImageModelName = Literal[
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "resnet18",
    "resnet34",
    "vgg16",
    "vgg16_bn",
]


def get_model_and_transforms(
    model_name: ImageModelName,
) -> Tuple[object, object]:
    """訓練済みの画像分類モデルと、モデルに対応した画像の前処理関数を取得する。

    Parameters
    ----------
    model_name : ImageModelName
        モデル名称。

    Returns
    -------
    model, transforms: Tuple[object, object]
        画像分類モデル、画像の前処理関数。

    Raises
    ------
    ValueError
        `model_name` に不適切なモデル名を指定した場合。
    """
    if model_name == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b0(weights=weights)
    elif model_name == "efficientnet_b1":
        weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2
        model = torchvision.models.efficientnet_b1(weights=weights)
    elif model_name == "efficientnet_b2":
        weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b2(weights=weights)
    elif model_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet34(weights=weights)
    elif model_name == "vgg16":
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        model = torchvision.models.vgg16(weights=weights)
    elif model_name == "vgg16_bn":
        weights = torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1
        model = torchvision.models.vgg16_bn(weights=weights)
    else:
        raise ValueError(f'"{model_name}" is invalid')
    return model, weights.transforms


class ImageVectorizer(torch.nn.Module):
    def __init__(self, model_name: ImageModelName):
        super(ImageVectorizer, self).__init__()
        model, transforms = get_model_and_transforms(model_name)
        self._transforms = transforms
        self.backborn = model

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        ndim = image_tensor.ndim
        if not 3 <= ndim <= 4:
            raise ValueError("`image_tensor` should be 3 or 4 dimensional torch.Tensor")
        elif ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        return self.backborn(image_tensor)

    @property
    def ndim(self) -> int:
        return 1000

    @property
    def transforms(self):
        return self._transforms()


class ImageClassifier(torch.nn.Module):
    def __init__(self, model_name: ImageModelName, n_classes: int = 2):
        super(ImageClassifier, self).__init__()
        self.vectorizer = ImageVectorizer(model_name)
        self.activation = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(1000, n_classes)
        self.n_classes = n_classes

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        vector = self.vectorizer(image_tensor)
        return self.classifier(self.activation(vector))

    @property
    def transforms(self):
        return self.vectorizer.transforms


if __name__ == "__main__":
    for model_name in (
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "resnet18",
        "resnet34",
        "vgg16",
        "vgg16_bn",
    ):
        model, transforms = get_model_and_transforms(model_name)
    try:
        get_model_and_transforms("hogehoge")
    except ValueError:
        pass

    for model_name in (
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "resnet18",
        "resnet34",
        "vgg16",
        "vgg16_bn",
    ):
        model = ImageClassifier(model_name)
        image = torch.Tensor(size=(3, 224, 224))
        model(image)
