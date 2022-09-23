import torchvision


def read_jpg(path: str):
    """jpg 画像の読み込み。

    Tutorial のコードから丸パクりしたもの。

    Parameters
    ----------
    path : str
        画像のファイルパス。

    Returns
    -------
    image_tensor: torch.Tensor
        画像のテンソル。
    """
    image_tensor = torchvision.io.read_image(path)
    if image_tensor.shape[0] == 1:
        # 1channel=白黒画像があるので3channelにconvertしています。
        image_tensor = image_tensor.expand(3, *image_tensor.shape[1:])
    return image_tensor
