"""
Utility Functions
=================
Shared utilities: sigmoid, relevance classification, model download.
"""

from __future__ import annotations

import os

import numpy as np
import requests


# ------------------------------------------------------------------
# Math helpers
# ------------------------------------------------------------------


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    return float(1 / (1 + np.exp(-np.clip(x, -500, 500))))


def get_relevance_level(score: float) -> str:
    """Map a normalized score in [0, 1] to a human-readable relevance label.

    Args:
        score: Sigmoid-normalized relevance score.

    Returns:
        One of ``"Highly relevant"``, ``"Moderately relevant"``,
        ``"Somewhat relevant"``, or ``"Low relevance"``.
    """
    if 0.8 <= score <= 1.0:
        return "Highly relevant"
    elif 0.5 <= score < 0.8:
        return "Moderately relevant"
    elif 0.2 <= score < 0.5:
        return "Somewhat relevant"
    else:
        return "Low relevance"


# ------------------------------------------------------------------
# Model download
# ------------------------------------------------------------------


def download_model_if_missing(local_path: str, model_url: str) -> str:
    """Download a model file if it does not exist locally.

    Uses atomic rename (``os.replace``) to prevent corruption from
    interrupted downloads.

    Args:
        local_path: Target file path on disk.
        model_url: Remote URL to download from.

    Returns:
        The *local_path* (whether it already existed or was just downloaded).
    """
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    parent_dir = os.path.dirname(local_path)
    os.makedirs(parent_dir, exist_ok=True)

    temp_path = local_path + ".tmp"
    print(f"\nDownloading from {model_url}")
    print(f"Saving to {local_path}")

    try:
        with requests.get(model_url, stream=True, timeout=(10, 300)) as r:
            r.raise_for_status()
            file_size = int(r.headers.get("content-length", 0))
            downloaded = 0

            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    _print_progress(downloaded, file_size)

        print()
        os.replace(temp_path, local_path)
        print(f"Download complete: {local_path}")

    except Exception as e:
        print()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

    return local_path


def _print_progress(current: int, total: int) -> None:
    """Print a console progress bar."""
    if total == 0:
        print(
            f"\rDownloaded: {current / 1024 / 1024:.2f} MB",
            end="",
            flush=True,
        )
        return

    percent = int(current * 100 / total)
    bar_length = 50
    filled = percent * bar_length // 100
    bar = "[" + "=" * filled + ">" + " " * (bar_length - filled - 1) + "]"
    current_mb = current / 1024 / 1024
    total_mb = total / 1024 / 1024
    print(
        f"\rProgress: {bar} {percent}% ({current_mb:.2f} / {total_mb:.2f} MB)",
        end="",
        flush=True,
    )
