"""
Load a dataset and compute for each sample the duration in seconds. We then save the average and std.
"""

from datasets import load_dataset, concatenate_datasets
from fleurs import LANG_TO_CONFIG_MAPPING
import fire
import numpy as np
import pandas as pd
from mozilla_cv import MozillaCVDataset, LANGS_TO_LOAD_REMOTE
import os
import librosa
from tqdm import tqdm
from joblib import Parallel, delayed
from transformers import AutoTokenizer
import string


def strip_punctuation(s):
    return s.translate(str.maketrans("", "", string.punctuation))


def count_tokens(text, tokenizer):
    return len(tokenizer.batch_decode(tokenizer(text)["input_ids"])) - 3


def compute_stats(
    reference: str,
    tokenizer,
    array: np.ndarray = None,
    sampling_rate: int = None,
    path: str = None,
):
    try:
        if path:
            arr, rs = librosa.load(path, sr=16000)
            sec = arr.shape[0] / 16000

        else:  # assuming array and sampling rate are not None
            sec = array.shape[0] / sampling_rate

        num_toknes = count_tokens(strip_punctuation(reference), tokenizer)

    except Exception as e:
        print(e)
        sec = np.nan
        num_toknes = np.nan

    return sec, num_toknes


def main(
    dataset: str,
    lang: str,
    output_dir: str,
    model: str,
    num_workers: int,
    reference_col: str,
):
    dataset_id = dataset.replace("/", "--")
    outfile = f"{output_dir}/stats_{dataset_id}_{lang}.csv"
    is_mozilla_local = "mozilla" in dataset and lang not in LANGS_TO_LOAD_REMOTE
    if os.path.exists(outfile):
        print(f"Out file {outfile} exists already. Skipping...")
        return

    tokenizer = AutoTokenizer.from_pretrained(model)

    results = list()

    if is_mozilla_local:
        cv_dataset = MozillaCVDataset(
            "/data/milanlp/attanasiog/fair_asr/cv-corpus-16.0-2023-12-06",
            lang,
            "all",
            decode_audio=True,
        )

        stats = Parallel(n_jobs=num_workers, verbose=1, batch_size=1000)(
            delayed(compute_stats)(
                row[reference_col],
                tokenizer,
                path=row["audio"],
            )
            for idx, row in cv_dataset.data.iterrows()
        )
        rids = [f"{row['split']}_{idx}" for idx, row in cv_dataset.data.iterrows()]

    else:
        dfs = list()
        for split in ["train", "validation", "test"]:
            lang_code = LANG_TO_CONFIG_MAPPING[lang] if "fleurs" in dataset else lang

            data = load_dataset(
                dataset,
                lang_code,
                split=split,
                num_proc=num_workers,
                trust_remote_code=True,
            )
            data = data.add_column("split", [split] * len(data))
            dfs.append(data)

        data = concatenate_datasets(dfs)

        stats = Parallel(n_jobs=num_workers, verbose=1, batch_size=1000)(
            delayed(compute_stats)(
                row[reference_col],
                tokenizer,
                array=row["audio"]["array"],
                sampling_rate=row["audio"]["sampling_rate"],
            )
            for row in data
        )

        rids = [f"{row['split']}_{idx}" for idx, row in enumerate(data)]

    df = pd.DataFrame(stats, columns=["duration", "num_tokens"], index=rids)

    df.to_csv(outfile)


if __name__ == "__main__":
    fire.Fire(main)
