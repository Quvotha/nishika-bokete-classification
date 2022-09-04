import argparse
import os
from typing import Iterator

import imagehash
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from scripts.log import get_logger


def _get_args():
    input_dir_default = os.path.join(
        os.path.expanduser("~"), "datasets", "nishika", "bokete"
    )
    fold_dir_default = os.path.join(os.path.abspath(os.path.dirname(__name__)), "fold")
    log_dir_default = os.path.join(os.path.abspath(os.path.dirname(__name__)), "log")
    parser = argparse.ArgumentParser(
        description="""This script make and save cross validation fold.
        Clustering imaegs by image-hash diff, and split data so that images in same cluster
        will be in validation set at same cv-iteration. Clustering result (clusters.csv) and
        data splitting result (cv.csv) will be saved."""
    )
    parser.add_argument(
        "-th",
        "--threshold",
        type=float,
        help="""Threshold of image hash difference. For an image, if the hash difference against the most
         similar image in an existing cluster is greater than the threshold, the image cannot be in that cluster.""",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default=input_dir_default,
        help=f'Directory where competition datasets are stored, by default "{input_dir_default}".',
    )
    parser.add_argument(
        "-f",
        "--fold_dir",
        type=str,
        default=fold_dir_default,
        help=f'Directory where result will be saved, by default "{fold_dir_default}".',
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        default=log_dir_default,
        help=f'Directory where log will be saved, by default "{log_dir_default}".',
    )
    args = parser.parse_args()
    return args


def gen_image_filepaths(
    input_dir: str, train: pd.DataFrame, test: pd.DataFrame
) -> Iterator[str]:
    """Yeild image filepath.

    Parameters
    ----------
    input_dir : str
        Directory where competition datasets are stored.
    train : pd.DataFrame
        train.csv.
    test : pd.DataFrame
        test.csv.

    Yields
    ------
    image_filepath : Iterator[str]
        Filepath to image.
    """
    for is_train in (True, False):
        if is_train:
            for filename in train["odai_photo_file_name"].to_numpy():
                yield os.path.join(input_dir, "train", filename)
        else:
            for filename in test["odai_photo_file_name"].to_numpy():
                yield os.path.join(input_dir, "test", filename)


def main(threshold: float, input_dir: str, fold_dir: str, log_dir: str) -> None:
    """Main process.

    Parameters
    ----------
    threshold : float
        Threshold of image hash difference, used for image clustering.
    input_dir : str
        Directory where competition datasets are stored.
    fold_dir : str
        Directory where result will be saved.
    log_dir : str
        Directory where log will be saved.
    """
    # Validation
    assert os.path.isdir(input_dir), f'"{input_dir}" dose not exist'
    assert os.path.isdir(fold_dir), f'"{fold_dir}" dose not exist'
    assert os.path.isdir(log_dir), f'"{log_dir}" dose not exist'
    assert (
        isinstance(threshold, float) and threshold > 0.0
    ), "`threshold` must be positive float"

    # Prepare logger
    logger = get_logger(os.path.join(log_dir, "make_fold.log"))
    logger.debug('input_dir: "{}"'.format(input_dir))
    logger.debug('fold_dir: "{}"'.format(fold_dir))
    logger.debug('log_dir: "{}"'.format(log_dir))
    logger.debug("threshold: {}".format(threshold))

    # Load dataset
    train = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test = pd.read_csv(os.path.join(input_dir, "test.csv"))

    # Ensure filename is unique and no-missing
    assert not train["odai_photo_file_name"].isnull().any()
    assert not train["odai_photo_file_name"].duplicated().any()
    assert not test["odai_photo_file_name"].isnull().any()
    assert not test["odai_photo_file_name"].duplicated().any()
    assert not train["odai_photo_file_name"].isin(test["odai_photo_file_name"]).any()

    # # Run image clustering
    clusters = []  # list of list[filename]
    hash_values = {}  # filename: image hash
    num_images = train.shape[0] + test.shape[0]
    logger.info("Start clustering {} images".format(num_images))
    for i, filepath in enumerate(gen_image_filepaths(input_dir, train, test)):
        # Cache image hash
        filename = os.path.basename(filepath)
        image = Image.open(filepath)
        hash_value = imagehash.phash(image)  # Slow
        hash_values[filename] = hash_value
        # Find or create a cluster where the image will be in
        if not clusters:
            clusters.append([filename])
        else:
            # Find nearest cluster
            distance_from_nearest_cluster = nearest_cluster_idx = None
            for cluster_idx, image_filenames_in_cluster in enumerate(clusters):
                distance_from_cluster = min(
                    [hash_value - hash_values[f] for f in image_filenames_in_cluster]
                )
                if cluster_idx == 0:
                    distance_from_nearest_cluster = distance_from_cluster
                    nearest_cluster_idx = cluster_idx
                elif distance_from_cluster < distance_from_nearest_cluster:
                    distance_from_nearest_cluster = distance_from_cluster
                    nearest_cluster_idx = cluster_idx
            # The image will be in the nearest cluster if image hash difference <= threshold, otherwise create new one
            if distance_from_nearest_cluster <= threshold:
                clusters[nearest_cluster_idx].append(filename)
            else:
                clusters.append([filename])
        # Show progress
        if (i + 1) % 1000 == 0 or i == num_images - 1:
            logger.info("Complete {}/{} files".format(i + 1, num_images))
    logger.info("Complete! {} clusters are created.".format(len(clusters)))
    del hash_values, test

    # Save clustering result
    clustering_result_ = []  # list of cluster_id, filename
    for cluster_id, image_filenames in enumerate(clusters):
        for filename in image_filenames:
            clustering_result_.append((cluster_id, filename))
    clustering_result = pd.DataFrame(
        clustering_result_, columns=["cluster_id", "odai_photo_file_name"]
    )
    clustering_result_filepath = os.path.join(fold_dir, "clusters.csv")
    clustering_result.to_csv(clustering_result_filepath, index=False)
    logger.info('Clustering result is saved in "{}"'.format(clustering_result_filepath))
    del clustering_result_

    # Split data
    cv = pd.merge(
        clustering_result, train[["odai_photo_file_name", "is_laugh"]]
    ).reset_index(drop=True)
    assert cv.shape[0] == train.shape[0]
    del clustering_result, train
    splitter = StratifiedGroupKFold(n_splits=5, random_state=293427, shuffle=True)
    cv["oof_fold"] = -1
    for fold, (_, idx_valid) in enumerate(
        splitter.split(cv, y=cv["is_laugh"], groups=cv["cluster_id"])
    ):
        cv.loc[idx_valid, "oof_fold"] = fold
    assert cv.groupby("cluster_id")["oof_fold"].nunique().max() == 1

    # Save data splitting result
    cv_filepath = os.path.join(fold_dir, "cv.csv")
    cv[["odai_photo_file_name", "oof_fold"]].to_csv(cv_filepath, index=False)
    logger.info('Data split result is saved in "{}"'.format(cv_filepath))

    # Complete!!
    logger.info("Complete all!")


if __name__ == "__main__":
    args = _get_args()
    main(args.threshold, args.input_dir, args.fold_dir, args.log_dir)
