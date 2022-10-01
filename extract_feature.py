import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts.log import get_logger
from scripts.sequence_models import SequenceClassifier, SequenceModelName
from scripts.texts import cleanse
from scripts.utils import collate_fn, NishikaBoketeDataset


def extract(
    model_name: SequenceModelName,
    model_filepath: str,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
) -> pd.DataFrame:
    # Prepare model
    model = SequenceClassifier(model_name)
    model.load_state_dict(torch.load(model_filepath))
    model.to(device)
    model.eval()

    # Extract feature and calculate probability
    ids = []  # id
    features = []  # テキストの埋め込み
    probabilities = []  # 分類確率
    for batch in dataloader:
        ids_, _, texts, _ = batch
        with torch.inference_mode():
            input_ = model.preprocess(texts).to(device)
            features_ = model.vectorizer(input_)
            probabilities_ = model.classifier(features_)
        ids += ids_
        features.append(features_.detach().to("cpu"))
        probabilities.append(probabilities_.detach().to("cpu"))
    features = torch.vstack(features).numpy()
    probabilities = torch.vstack(probabilities).numpy()
    df = pd.DataFrame(data=features)
    df["id"] = ids
    df["Probability"] = probabilities
    return df.set_index("id").sort_index()


def _get_args() -> argparse.Namespace:
    input_dir_default = os.path.join(
        os.path.expanduser("~"), "datasets", "nishika", "bokete"
    )
    output_dir_default = os.path.join(
        os.path.abspath(os.path.dirname(__name__)), "data"
    )
    image_dir_default = os.path.join(input_dir_default, "train")
    log_dir_default = os.path.join(os.path.abspath(os.path.dirname(__name__)), "log")
    parser = argparse.ArgumentParser(
        description="Extract text embedding and classification probability by fine-tuned sequence model."
    )
    parser.add_argument(
        "filepath", type=str, help='Filepath to "train.csv" or "test.csv."'
    )
    parser.add_argument("model_name", type=str, help="Model name.")
    parser.add_argument(
        "model_filepath", type=str, help="Filepath to fine-tuned model weight."
    )
    parser.add_argument("batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=output_dir_default,
        help=f'Extracted feature will be saved at "<output_dir>/feature.csv". By default "{output_dir_default}."',
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
        "-nw",
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help=f"Number of processes passed to pytorch's dataloader, by default {os.cpu_count()}.",
    )
    args = parser.parse_args()
    return args


def main(
    filepath: str,
    model_name: SequenceModelName,
    model_filepath: str,
    batch_size: int,
    output_dir: str,
    image_dir: str,
    log_dir: str,
    num_workers: int,
):
    # Validation
    assert os.path.isfile(filepath), f'"{filepath}" dose not exist'
    assert os.path.isfile(model_filepath), f'"{model_filepath}" dose not exist'
    assert os.path.isdir(output_dir), f'"{output_dir}" dose not exist'
    assert os.path.isdir(image_dir), f'"{image_dir}" dose not exist'
    assert os.path.isdir(log_dir), f'"{log_dir}" dose not exist'
    assert (
        0 <= num_workers <= os.cpu_count()
    ), f"`num_workers` is out of range ({num_workers}, {os.cpu_count()})"

    # Get logger
    logger = get_logger(os.path.join(log_dir, "extract_feature.log"), __name__)

    # Debug
    logger.debug('filepath: "{}"'.format(filepath))
    logger.debug('model_name: "{}"'.format(model_name))
    logger.debug('model_filepath: "{}"'.format(model_filepath))
    logger.debug('batch_size: "{}"'.format(batch_size))
    logger.debug('output_dir: "{}"'.format(output_dir))
    logger.debug('image_dir: "{}"'.format(image_dir))
    logger.debug('log_dir: "{}"'.format(log_dir))
    logger.debug('num_workers: "{}"'.format(num_workers))

    # Load dataset
    data = pd.read_csv(filepath)

    # Preprocess text
    data["text"] = np.vectorize(cleanse)(data["text"])

    # Select device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Train model on {}".format(device_name))
    device = torch.device(device_name)

    # Extract text embedding
    feature = extract(
        model_name,
        model_filepath,
        device,
        torch.utils.data.DataLoader(
            NishikaBoketeDataset(data, image_dir),
            batch_size=batch_size,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        ),
    )
    feature.to_csv(os.path.join(output_dir, "feature.csv"))
    logger.info("Complete")


if __name__ == "__main__":
    args = _get_args()
    main(
        args.filepath,
        args.model_name,
        args.model_filepath,
        args.batch_size,
        args.output_dir,
        args.image_dir,
        args.log_dir,
        args.num_workers,
    )
