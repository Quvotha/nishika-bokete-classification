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
    input_dir_default = os.path.join(
        os.path.expanduser("~"), "datasets", "nishika", "bokete"
    )
    cv_filepath_default = os.path.join(
        os.path.abspath(os.path.dirname(__name__)), "fold", "cv.csv"
    )
    model_dir_default = os.path.join(
        os.path.abspath(os.path.dirname(__name__)), "models"
    )
    image_dir_default = os.path.join(input_dir_default, "train")
    log_dir_default = os.path.join(os.path.abspath(os.path.dirname(__name__)), "log")
    parser = argparse.ArgumentParser(description="Fine tune state-of-art image model.")
    parser.add_argument("image_model_name", type=str, help="Image model name.")
    parser.add_argument("sequence_model_name", type=str, help="Sequence model name.")
    parser.add_argument("n_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("lr", type=float, help="Learning rate.")
    parser.add_argument("batch_size", type=int, help="Batch size.")
    parser.add_argument("oof_fold", type=int, help="CV-iteration.")
    parser.add_argument(
        "-model",
        "--model_dir",
        type=str,
        default=model_dir_default,
        help="Directory where model will be saved.",
    )
    parser.add_argument(
        "-cv",
        "--cv_filepath",
        type=str,
        default=cv_filepath_default,
        help=f"""Path to cv filepath, by default "{cv_filepath_default}".
        This file is expected to be output by make_fold.py.""",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default=input_dir_default,
        help=f'Directory where competition datasets are stored, by default "{input_dir_default}".',
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
    parser.add_argument(
        "-w1",
        "--image_model_weight",
        type=str,
        default="",
        help="Path to pretrained image model weight.",
    )
    parser.add_argument(
        "-w2",
        "--sequence_model_weight",
        type=str,
        default="",
        help="Path to pretrained sequence model weight.",
    )
    parser.add_argument(
        "--freeze_image_vectorizer",
        action="store_true",
        help="If True, ImageVectorizer's weight will be NOT update during training.",
    )
    parser.add_argument(
        "--freeze_sequence_vectorizer",
        action="store_true",
        help="If True, SequenceVectorizer's weight will be NOT update during training.",
    )
    args = parser.parse_args()
    return args


def main(
    image_model_name: ImageModelName,
    sequence_model_name: SequenceModelName,
    n_epochs: int,
    lr: float,
    batch_size: int,
    oof_fold: int,
    cv_filepath: str,
    input_dir: str,
    image_dir: str,
    log_dir: str,
    model_dir: str,
    seed: int,
    num_workers: int,
    image_model_weight: str,
    sequence_model_weight: str,
    freeze_image_vectorizer: bool,
    freeze_sequence_vectorizer: bool,
):
    # Validation
    assert (
        isinstance(n_epochs, int) and n_epochs >= 1
    ), f"`epoch` should be interger >= 1 but {n_epochs} was given"
    assert (
        isinstance(lr, float) and 0.0 < lr
    ), f"`lr` should be positive float but {lr} was given"
    assert (
        isinstance(batch_size, int) and batch_size > 0
    ), f"`batch_size` should be positive interger but {batch_size} was given"
    assert (
        isinstance(oof_fold, int) and oof_fold >= 0
    ), f"`n_fold` should be non negative interger but {oof_fold} was given"
    assert os.path.isfile(cv_filepath), f'"{cv_filepath}" dose not exist'
    assert os.path.isdir(input_dir), f'"{input_dir}" dose not exist'
    assert os.path.isdir(image_dir), f'"{image_dir}" dose not exist'
    assert os.path.isdir(log_dir), f'"{log_dir}" dose not exist'
    assert os.path.isdir(model_dir), f'"{model_dir}" dose not exist'
    assert (
        isinstance(seed, int) and seed >= 0
    ), f"`seed` should be non-negative integer but {seed} was given"
    assert (
        0 <= num_workers <= os.cpu_count()
    ), f"`num_workers` is out of range ({num_workers}, {os.cpu_count()})"
    assert not (image_model_weight) or os.path.isfile(
        image_model_weight
    ), f"{image_model_weight} should be blank str or filepath of pretrained model"
    assert not (sequence_model_weight) or os.path.isfile(
        sequence_model_weight
    ), f"{sequence_model_weight} should be blank str or filepath of pretrained model"

    # Get logger
    logger = get_logger(os.path.join(log_dir, "finetune_model2.log"), __name__)

    # Debug
    logger.debug('image_model_name: "{}"'.format(image_model_name))
    logger.debug('sequence_model_name: "{}"'.format(sequence_model_name))
    logger.debug('n_epochs: "{}"'.format(n_epochs))
    logger.debug('lr: "{}"'.format(lr))
    logger.debug('off_fold: "{}"'.format(oof_fold))
    logger.debug('cv_filepath: "{}"'.format(cv_filepath))
    logger.debug('input_dir: "{}"'.format(input_dir))
    logger.debug('image_dir: "{}"'.format(image_dir))
    logger.debug('log_dir: "{}"'.format(log_dir))
    logger.debug('model_dir: "{}"'.format(model_dir))
    logger.debug('image_model_weight: "{}"'.format(image_model_weight))
    logger.debug('sequence_model_weight: "{}"'.format(sequence_model_weight))
    logger.debug('freeze_image_vectorizer: "{}"'.format(freeze_image_vectorizer))
    logger.debug('freeze_sequence_vectorizer: "{}"'.format(freeze_sequence_vectorizer))

    # Load dataset
    data = pd.read_csv(os.path.join(input_dir, "train.csv"))
    cv = pd.read_csv(cv_filepath)

    # Preprocess text
    data["text"] = np.vectorize(cleanse)(data["text"])

    # Split data into training/validation set
    train, valid = split_into_training_validation(data, cv, oof_fold)
    logger.debug(
        "Training set: {} rows, Validation set: {} rows".format(
            train.shape[0], valid.shape[0]
        )
    )
    assert (
        min(train.shape[0], valid.shape[0]) > 0
    ), f"At least one of training/validation set has not row {(train.shape[0], valid.shape[0], oof_fold)}"
    del data, cv

    # Make model output directory
    model_output_dir = os.path.join(
        model_dir,
        image_model_name + "+" + sequence_model_name.replace("/", "-"),
        f"fold{oof_fold}",
    )
    if not os.path.isdir(model_output_dir):
        logger.info('Make "{}" for saving model'.format(model_output_dir))
        os.makedirs(model_output_dir)

    # Set seed for reproducibity
    set_seeds(seed)

    # Select device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Train model on {}".format(device_name))
    device = torch.device(device_name)

    # Create model to be trained, use pretrained image/sequence model's weight if filepath is given
    model = BoketeClassifier(image_model_name, sequence_model_name)
    if image_model_weight:
        # Load pretrained image model's weight if filepath is given
        logger.info(
            'Load pretrained image model weight from "{}"'.format(image_model_weight)
        )
        image_model = ImageClassifier(image_model_name)
        image_model.load_state_dict(torch.load(image_model_weight))
        model.image_vectorizer.load_state_dict(
            image_model.vectorizer.state_dict().copy()
        )
        del image_model
    if sequence_model_weight:
        # Load pretrained sequence model's weight if filepath is given
        logger.info(
            'Load pretrained sequence model weight from "{}"'.format(
                sequence_model_weight
            )
        )
        sequence_model = SequenceClassifier(sequence_model_name)
        sequence_model.load_state_dict(torch.load(sequence_model_weight))
        model.sequence_vectorizer.load_state_dict(
            sequence_model.vectorizer.state_dict().copy()
        )
        del sequence_model
    model.to(device)

    # Prepare optimizer, scheduler, criterion and dataloaders
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=n_epochs, eta_min=0, last_epoch=-1
    # )
    criterion = torch.nn.BCEWithLogitsLoss()
    dataloader_train = torch.utils.data.DataLoader(
        NishikaBoketeDataset(train, image_dir),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    dataloader_valid = torch.utils.data.DataLoader(
        NishikaBoketeDataset(valid, image_dir),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    # Training/evaluation loop
    for epoch in range(1, n_epochs + 1):  # 1 epoch
        logger.info("Start {}/{} epoch".format(epoch, n_epochs))

        # Training
        model.train()
        if freeze_image_vectorizer:
            model.image_vectorizer.eval()
        if freeze_sequence_vectorizer:
            model.sequence_vectorizer.eval()
        total_loss_train = 0.0
        n_train = 0
        for batch in dataloader_train:  # 1 batch (training set)
            _, image_tensors, texts, labels = batch
            labels = torch.Tensor(labels).to(device)
            optimizer.zero_grad()
            images_transformed, tokenized = model.preprocess(image_tensors, texts)
            output = model(images_transformed.to(device), tokenized.to(device))
            loss = criterion(output[:, -1], labels)
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()
            n_train += len(output)
            del output, labels, batch, image_tensors, images_transformed, tokenized
            torch.cuda.empty_cache()
        # scheduler.step()

        # Save model
        torch.save(
            model.state_dict(),
            os.path.join(model_output_dir, f"epoch{str(epoch).zfill(4)}"),
        )

        # Evaluation
        model.eval()
        total_loss_valid = 0.0
        n_valid = 0
        for batch in dataloader_valid:  # 1 batch (validation set)
            _, image_tensors, texts, labels = batch
            labels = torch.Tensor(labels).to(device)
            images_transformed, tokenized = model.preprocess(image_tensors, texts)
            with torch.inference_mode():
                output = model(images_transformed.to(device), tokenized.to(device))
            loss = criterion(output[:, -1], labels)
            total_loss_valid += loss.item()
            n_valid += len(output)
            del output, labels, batch, image_tensors, images_transformed, tokenized
            torch.cuda.empty_cache()

        # Logging
        logger.info(
            "Epoch {}: Training loss = {} Validation loss = {}".format(
                epoch, total_loss_train / n_train, total_loss_valid / n_valid
            )
        )

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
