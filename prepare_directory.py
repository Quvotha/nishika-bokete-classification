import argparse
import os


def _get_args():
    root_dir_default = os.path.abspath(os.path.dirname(__name__))
    parser = argparse.ArgumentParser(
        description="This script make directories where outputs wlil be stored."
    )
    parser.add_argument(
        "-r",
        "--root_dir",
        type=str,
        default=root_dir_default,
        help=f'Top directory, by default "{root_dir_default}".',
    )
    args = parser.parse_args()
    return args


def main(root_dir: str):
    assert isinstance(
        root_dir, str
    ), f'`root_dir` must be str but "{root_dir}" was given.'
    assert os.path.isdir(root_dir), f'"{root_dir}" dose not exist.'
    for directory in ("models", "data", "fold", "log", "submissions"):
        os.makedirs(os.path.join(root_dir, directory), exist_ok=True)


if __name__ == "__main__":
    args = _get_args()
    main(args.root_dir)
