import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts.image_models import ImageClassifier, ImageModelName
from scripts.log import get_logger
from scripts.models import BoketeClassifier
from scripts.sequence_models import SequenceClassifier, SequenceModelName
from scripts.texts import cleanse
from scripts.utils import (
    collate_fn,
    NishikaBoketeDataset,
    set_seeds,
    split_into_training_validation,
)


def _get_args() -> argparse.Namespace:
    # 入力ファイル、出力ファイル、モデル、バッチサイズ、ワーカー数、あればサブミッションのテンプレ
    input_filepath_default = os.path.join(
        os.path.expanduser("~"), "datasets", "nishika", "bokete", "test.csv"
    )
    image_dir_default = os.path.join(os.path.dirname(input_filepath_default), "test")
    log_dir_default = os.path.join(os.path.abspath(os.path.dirname(__name__)), "log")
    parser = argparse.ArgumentParser(
        description="Infere bokete label by finetuned model."
    )
    parser.add_argument(
        "output_filepath",
        type=str,
        help="Filepath where inference result will be written.",
    )
    parser.add_argument("image_model_name", type=str, help="Image model name.")
    parser.add_argument("sequence_model_name", type=str, help="Sequence model name.")
    parser.add_argument("image_model_weight", type=str)
    parser.add_argument("sequemce_model_weight", type=str)
    parser.add_argument("batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "-i",
        "--input_filepath",
        type=str,
        default=input_filepath_default,
        help=f'Input filepath, by default "{input_filepath_default}".',
    )
    parser.add_argument(
        "-im",
        "--image_dir",
        type=str,
        default=image_dir_default,
        help=f'Directory where competition image files are stored, by default "{image_dir_default}".',
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        default=log_dir_default,
        help=f'Directory where log will be saved, by default "{log_dir_default}".',
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed, by default 42.",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help=f"Number of processes passed to pytorch's dataloader, by default {os.cpu_count()}.",
    )
    args = parser.parse_args()
    return args


def main(
    input_filepath: str,
    output_filepath: str,
    image_model_name: ImageModelName,
    sequence_model_name: SequenceModelName,
    image_model_weight: str,
    sequence_model_weight: str,
    batch_size: int,
    image_dir: str,
    log_dir: str,
    seed: int,
    num_workers: int,
):
    # Validation
    assert (
        isinstance(batch_size, int) and batch_size > 0
    ), f"`batch_size` should be positive interger but {batch_size} was given"
    assert os.path.isfile(input_filepath), f'"{input_filepath}" dose not exist'
    assert os.path.isdir(image_dir), f'"{image_dir}" dose not exist'
    assert os.path.isdir(log_dir), f'"{log_dir}" dose not exist'
    assert (
        isinstance(seed, int) and seed >= 0
    ), f"`seed` should be non-negative integer but {seed} was given"
    assert (
        0 <= num_workers <= os.cpu_count()
    ), f"`num_workers` is out of range ({num_workers}, {os.cpu_count()})"
    assert os.path.isfile(image_model_weight), f'"{image_model_weight}" dose not exist'
    assert os.path.isfile(
        sequence_model_weight
    ), f'"{sequence_model_weight}" dose not exist'

    # Get logger
    logger = get_logger(os.path.join(log_dir, "inference.log"), __name__)

    # Debug
    logger.debug('input_filepath: "{}"'.format(input_filepath))
    logger.debug('output_filepath: "{}"'.format(output_filepath))
    logger.debug('image_model_name: "{}"'.format(image_model_name))
    logger.debug('sequence_model_name: "{}"'.format(sequence_model_name))
    logger.debug('image_model_weight: "{}"'.format(image_model_weight))
    logger.debug('sequence_model_weight: "{}"'.format(sequence_model_weight))
    logger.debug('batch_size: "{}"'.format(batch_size))
    logger.debug('image_dir: "{}"'.format(image_dir))
    logger.debug('seed: "{}"'.format(seed))
    logger.debug('num_workers: "{}"'.format(num_workers))

    # Load dataset
    data = pd.read_csv(input_filepath)

    # Preprocess text
    data["text"] = np.vectorize(cleanse)(data["text"])

    # Set seed for reproducibity
    set_seeds(seed)

    # Select device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Train model on {}".format(device_name))
    device = torch.device(device_name)

    # Create model to be trained, use pretrained image/sequence model's weight if filepath is given
    model = BoketeClassifier(image_model_name, sequence_model_name)
    image_model = ImageClassifier(image_model_name)
    image_model.load_state_dict(torch.load(image_model_weight))
    model.image_vectorizer.load_state_dict(image_model.vectorizer.state_dict().copy())
    sequence_model = SequenceClassifier(sequence_model_name)
    sequence_model.load_state_dict(torch.load(sequence_model_weight))
    model.sequence_vectorizer.load_state_dict(
        sequence_model.vectorizer.state_dict().copy()
    )
    del image_model, sequence_model
    model.to(device)
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        NishikaBoketeDataset(data, image_dir),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    ids = []
    prediction = []
    for batch in dataloader:
        ids_, image_tensors, texts, _ = batch
        images_transformed, tokenized = model.preprocess(image_tensors, texts)
        output = model(images_transformed.to(device), tokenized.to(device))
        ids += ids_
        prediction.append(output.sigmoid().detach().to("cpu").numpy())
        del image_tensors, images_transformed, tokenized
    result = pd.DataFrame(data=np.vstack(prediction), index=ids)
    result.to_csv(output_filepath)

    logger.info("Complete")


if __name__ == "__main__":
    args = _get_args()
    main(
        args.image_model_name,
        args.sequence_model_name,
        args.n_epochs,
        args.lr,
        args.batch_size,
        args.oof_fold,
        args.cv_filepath,
        args.input_dir,
        args.image_dir,
        args.log_dir,
        args.model_dir,
        args.seed,
        args.num_workers,
        args.image_model_weight,
        args.sequence_model_weight,
        args.freeze_image_vectorizer,
        args.freeze_sequence_vectorizer,
    )
