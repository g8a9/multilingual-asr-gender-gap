import json
import os
import time

import fire
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from utils import MozillaCVDataset


def main(config_file: str, config_id: int, output_dir, num_proc: int = 18):
    with open(config_file) as fp:
        configs = json.load(fp)

    config = configs[str(config_id)]
    print("Loaded config:")
    print(config)

    lang = config["lang"]
    dataset = config["dataset"]
    split = config["split"]
    dataset_id = dataset.replace("/", "--")

    dataset = MozillaCVDataset(
        "/data/milanlp/attanasiog/fair_asr/cv-corpus-16.0-2023-12-06",
        lang,
        split,
        decode_audio=True,
    )
    dataset.validate_audio(n_jobs=num_proc)
    raw_data = dataset.data.loc[dataset.data["is_valid"] == True]

    os.makedirs(output_dir, exist_ok=True)

    # if split == "all":
    for split in tqdm(["train", "validation", "test"], desc="Split"):
        
        # current_data = raw_data[split]
        # if not isinstance(current_data, pd.DataFrame):
            # df = current_data.to_pandas()
        # else:
        df = raw_data.loc[raw_data["split"] == split]

        def compute_stats(values, column):
            stats = list()
            for g in values:
                cdf = df.loc[df[column] == g]
                user_count = cdf.client_id.unique().size

                if user_count > 1:
                    value_counts = cdf.client_id.value_counts().values
                    spu_stats = {
                        "median_spu": np.median(value_counts),
                        "mean_spu": value_counts.mean(),
                        "std_spu": value_counts.std(),
                        "90p_spu": np.percentile(value_counts, 90),
                        "75p_spu": np.percentile(value_counts, 75),
                        "50p_spu": np.percentile(value_counts, 50),
                        "25p_spu": np.percentile(value_counts, 25),
                        "max_spu": np.max(value_counts),
                        "min_spu": np.min(value_counts),
                    }
                else:
                    spu_stats = {
                        "median_spu": np.nan,
                        "mean_spu": np.nan,
                        "std_spu": np.nan,
                        "90p_spu": np.nan,
                        "75p_spu": np.nan,
                        "50p_spu": np.nan,
                        "25p_spu": np.nan,
                        "max_spu": np.nan,
                        "min_spu": np.nan,
                    }

                stats.append(
                    {
                        "dataset": dataset_id,
                        "lang": lang,
                        "split": split,
                        column: g,
                        "users": user_count,
                        "sample_count": len(cdf),
                        **spu_stats,
                    }
                )

            return pd.DataFrame(stats)

        # 1. Statistics stratified by Gender
        gender_stats = compute_stats(["male", "female", "other"], "gender")
        gender_stats.to_csv(
            f"{output_dir}/{dataset_id}_{lang}_{split}_gender_stats.csv", index=None
        )

        # 2. Statistics stratified by Age
        gender_stats = compute_stats(
            [
                "teens",
                "twenties",
                "thirties",
                "fourties",
                "fifties",
                "sixties",
                "seventies",
                "eighties",
                "nineties",
            ],
            "age",
        )
        output_file = f"{dataset_id}_{lang}_{split}_age_stats.csv"
        gender_stats.to_csv(f"{output_dir}/{output_file}", index=None)


if __name__ == "__main__":
    stime = time.time()
    print("STARTED")
    fire.Fire(main)
    print(f"FINISHED after {int(time.time() - stime)}")
