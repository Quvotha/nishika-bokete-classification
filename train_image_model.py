import argparse
import os

import pandas as pd
import torch

from scripts.image_models import ImageClassifier
from scripts.log import get_logger
from scripts.utils import collate_fn, NishikaBoketeDataset, set_seeds


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
    parser.add_argument("model_name", type=str, help="Model name.")
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
    args = parser.parse_args()
    return args


def main(
    model_name: str,
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
    # Prepare logger
    logger = get_logger(os.path.join(log_dir, "train_image_model.log"), __name__)
    logger.debug('model_name: "{}"'.format(model_name))
    logger.debug('n_epochs: "{}"'.format(n_epochs))
    logger.debug('lr: "{}"'.format(lr))
    logger.debug('off_fold: "{}"'.format(oof_fold))
    logger.debug('cv_filepath: "{}"'.format(cv_filepath))
    logger.debug('input_dir: "{}"'.format(input_dir))
    logger.debug('image_dir: "{}"'.format(image_dir))
    logger.debug('log_dir: "{}"'.format(log_dir))
    logger.debug('model_dir: "{}"'.format(model_dir))

    # Load dataset
    data = pd.read_csv(os.path.join(input_dir, "train.csv"))
    cv = pd.read_csv(cv_filepath)

    # Split data into training/validation set
    oof_filenames = cv.query(f"oof_fold == {oof_fold}")["odai_photo_file_name"]
    mask = data["odai_photo_file_name"].isin(oof_filenames)  # True: validation set
    train = data.loc[~mask]
    valid = data.loc[mask]
    logger.debug(
        "Training set: {} rows, Validation set: {} rows".format(
            train.shape[0], valid.shape[0]
        )
    )
    assert min(train.shape[0], valid.shape[0]) > 0

    # Prepare model output directory
    model_output_dir = os.path.join(model_dir, model_name, f"fold{oof_fold}")
    if not os.path.isdir(model_output_dir):
        logger.info('Make "{}" for saving model'.format(model_output_dir))
        os.makedirs(model_output_dir)

    # Train and evaluate model
    set_seeds(seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Train model on {}".format(device_name))
    device = torch.device(device_name)
    model = ImageClassifier(model_name).to(device)
    image_transforms = model.transforms
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=0, last_epoch=-1
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    dataset_train = NishikaBoketeDataset(train, image_dir)
    dataset_valid = NishikaBoketeDataset(valid, image_dir)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    for epoch in range(1, n_epochs + 1):
        logger.info("Start {}/{} epoch".format(epoch, n_epochs))

        # Training
        model.train()
        total_loss_train = 0.0
        n_train = 0
        for batch in dataloader_train:
            _, images, _, labels = batch
            images_transformed = torch.stack(
                [image_transforms(image) for image in images]
            ).to(device)
            # images = images.to(device)
            labels = torch.Tensor(labels).to(device)
            optimizer.zero_grad()
            output = model(images_transformed)  # shape = (batch_size, 1)
            loss = criterion(output[:, -1], labels)
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()
            n_train += len(output)
            del images_transformed, output, labels, batch
            torch.cuda.empty_cache()
        scheduler.step()

        # Save model
        torch.save(
            model.state_dict(),
            os.path.join(model_output_dir, f"epoch{str(epoch).zfill(4)}"),
        )

        # Evaluation
        model.eval()
        total_loss_valid = 0.0
        n_valid = 0
        for batch in dataloader_valid:
            _, images, _, labels = batch
            images_transformed = torch.stack(
                [image_transforms(image) for image in images]
            ).to(device)
            # images = images.to(device)
            labels = torch.Tensor(labels).to(device)
            with torch.inference_mode():
                output = model(images_transformed)
            loss = criterion(output[:, -1], labels)
            total_loss_valid += loss.item()
            n_valid += len(output)
            del images_transformed, output, labels, batch
            torch.cuda.empty_cache()
        logger.info(
            "Epoch {}: Training loss = {} Validation loss = {}".format(
                epoch, total_loss_train / n_train, total_loss_valid / n_valid
            )
        )

    logger.info("Complete")


if __name__ == "__main__":
    args = _get_args()
    main(
        args.model_name,
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
    )