import os
import numpy as np
import fire
import json
import time
import torch
from fleurs import LANG_TO_CONFIG_MAPPING
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import set_seed
import librosa
from transcriber import SimpleTranscriber
from mozilla_cv import MozillaCVDataset
import math
from datasets import load_dataset, concatenate_datasets, Audio


TARGET_SAMPLING_RATE = 16_000
MAX_LENGTH = 30


def generate_splits(
    df,
    target_col,
    majority_group,
    minority_group,
    minority_accept_percentile,
    speaker_id_col,
    reference_col,
    subsample_frac: float = 1,
    subsample_n: int = -1,
):
    """
    Some references might be empty due to issues in the original dataset. We filter them out.
    """
    print("Len before filtering rows with empty reference:", len(df))
    df = df.loc[~df[reference_col].isna()]
    print("Len after:", len(df))

    # print(f"Empty transcriptions found:", len(df.loc[df["transcription"].isna()]))
    # df["transcription"] = df["transcription"].fillna("")

    minority_df = df.loc[df[target_col] == minority_group]
    print(f"Unique minority speakers count: {minority_df.client_id.unique().size}")
    majority_df = df.loc[df[target_col] == majority_group]
    print(f"Unique majority speakers count: {majority_df.client_id.unique().size}")

    # def find_users_below_percentile(df):
    #     spu_array = df[speaker_id_col].value_counts()
    #     perc_abs_threshold = np.percentile(spu_array.values, minority_accept_percentile)
    #     selected_users = spu_array.loc[spu_array <= perc_abs_threshold]
    #     return set(selected_users.index)

    # # 1. select users based on SPU percentile
    # mino_users = find_users_below_percentile(minority_df)
    # print(f"Selected {len(mino_users)} users from the minority group")
    # majo_users = find_users_below_percentile(majority_df)
    # print(f"Selected {len(majo_users)} users from the majority group")

    # minority_df = minority_df.loc[minority_df[speaker_id_col].isin(mino_users)]
    # majority_df = majority_df.loc[majority_df[speaker_id_col].isin(majo_users)]
    # print(f"{len(minority_df)} records from minority users")
    # print(f"{len(majority_df)} records from majority users")

    # compute relative user frequency, needed for stratification
    def add_frequency_weight(df):
        w = df[speaker_id_col].value_counts(normalize=True)
        w.name = "weight"
        return df.join(w, on=speaker_id_col)

    minority_df, majority_df = map(add_frequency_weight, (minority_df, majority_df))
    min_count, maj_count = len(minority_df), len(majority_df)

    results = dict()
    results["largest_group"] = "minority" if min_count > maj_count else "majority"

    # def add_prefix_to_keys(d, prefix):
    #     return {f"{prefix}_{k}": v for k, v in d.items()}

    overall_maj = majority_df.sample(n=min(min_count, maj_count), weights="weight")
    overall_min = minority_df.sample(n=min(min_count, maj_count), weights="weight")

    full_df = pd.concat([overall_maj, overall_min])

    if subsample_frac != 1.0:
        full_df = full_df.sample(frac=subsample_frac, weights="weight")

    if subsample_n != -1 and subsample_n < len(full_df):
        full_df = full_df.sample(n=subsample_n, weights="weight")

    user_ids = full_df[speaker_id_col].unique()
    user_genders = [
        full_df.loc[full_df[speaker_id_col] == uid][target_col].iloc[0]
        for uid in user_ids
    ]

    # 3. split them in train and test
    train_users, test_users = train_test_split(
        user_ids, train_size=0.8, shuffle=True, stratify=user_genders
    )

    train_df = full_df.loc[full_df[speaker_id_col].isin(train_users)]
    test_df = full_df.loc[full_df[speaker_id_col].isin(test_users)]

    return train_df, test_df


def main(
    # config_file: str,
    # config_id: int,
    model: str,
    dataset: str,
    lang: str,
    output_dir: str,
    load_type: str,
    batch_size: int = 1,
    num_workers: int = 1,
    overwrite_output: bool = False,
    # enable_chunk_decoding: bool = False,  # set to True to prevent OOM
    # chunk_size: int = 3000,
    # column names to change depending on the dataset used
    target_col: str = "gender",
    reference_col: str = "sentence",
    speaker_id_col: str = "client_id",
    minority_group: str = "female",
    majority_group: str = "male",
    minority_accept_percentile: float = 99,
    # args for the compression training set
    subsample_frac: float = 1.0,
    subsample_n: int = -1,
    # num_users: int = 10,
    # num_samples_per_user: int = 100,
):
  
    clean_model_name = model.replace("/", "--")
    clean_dataset_name = dataset.replace("/", "--")
    # local_dir = config.get("local_dir", None)

    train_out_file = f"train_{clean_model_name}_{clean_dataset_name}_{lang}.csv"
    if os.path.exists(f"{output_dir}/{train_out_file}") and not overwrite_output:
        print(f"Output file {train_out_file} exists already. Skipping...")
        return

    set_seed(42)

    if load_type == "remote":
        dev_data = load_dataset(
            dataset,
            lang,
            split="validation",
            num_proc=num_workers,
            trust_remote_code=True,
        )
        dev_data = dev_data.add_column("split", ["validation"] * len(dev_data))
        test_data = load_dataset(
            dataset, lang, split="test", num_proc=num_workers, trust_remote_code=True
        )
        test_data = test_data.add_column("split", ["test"] * len(test_data))
        data = concatenate_datasets([dev_data, test_data])

        data = data.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))

        raw_audios = [d["array"] for d in data["audio"]]
        data = data.remove_columns(["audio"])
        data = data.add_column("raw_audio", raw_audios)
        df = data.to_pandas()
        print("Loading finished.")
    elif load_type == "local":
        dataset = MozillaCVDataset(
            "/data/milanlp/attanasiog/fair_asr/cv-corpus-16.0-2023-12-06",
            lang,
            "all",
            decode_audio=True,
        )  # type: ignore
        dataset.validate_audio(n_jobs=num_workers)
        print("Validity check:", dataset.data["is_valid"].value_counts())
        raw_data = dataset.data.loc[dataset.data["is_valid"] == True]

        df = raw_data
        df = df.loc[df["split"] != "train"]
        print("Loading finished.")
    else:
        raise ValueError("Load type unknown")

    # 2. Generate training and test df
    train_df, test_df = generate_splits(
        df,
        target_col=target_col,
        minority_group=minority_group,
        majority_group=majority_group,
        minority_accept_percentile=minority_accept_percentile,
        speaker_id_col=speaker_id_col,
        reference_col=reference_col,
        subsample_frac=subsample_frac,
        subsample_n=subsample_n,
    )

    print("Train shape", train_df.shape)
    print("Test shape", test_df.shape)

    # 4. get embeddings
    transcriber = SimpleTranscriber(
        model_name_or_path=model,
        tgt_lang=lang,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )

    if load_type == "local":

        def load(path):
            arr, sr = librosa.load(path, sr=TARGET_SAMPLING_RATE)
            return arr

        train_df["raw_audio"] = train_df["audio"].apply(load)
        test_df["raw_audio"] = test_df["audio"].apply(load)

    train_df["duration"] = train_df.apply(
        lambda row: len(row["raw_audio"]) / 16000, axis=1
    )
    test_df["duration"] = test_df.apply(
        lambda row: len(row["raw_audio"]) / 16000, axis=1
    )
    train_df["idx_last_content"] = train_df.apply(
        lambda row: math.ceil(row["duration"] / 0.02), axis=1
    )
    test_df["idx_last_content"] = test_df.apply(
        lambda row: math.ceil(row["duration"] / 0.02), axis=1
    )

    train_arrays = train_df["raw_audio"].tolist()
    test_arrays = test_df["raw_audio"].tolist()

    print(len(train_arrays), train_arrays[0].shape)

    train_outputs = transcriber.forward_encoder(
        raw_audio=train_arrays,
        sampling_rate=TARGET_SAMPLING_RATE,
        output_hidden_states=True,
        show_progress_bar=True,
        batch_size=batch_size,
        max_length=30 * TARGET_SAMPLING_RATE,
    )
    test_outputs = transcriber.forward_encoder(
        raw_audio=test_arrays,
        sampling_rate=TARGET_SAMPLING_RATE,
        output_hidden_states=True,
        show_progress_bar=True,
        batch_size=batch_size,
        max_length=30 * TARGET_SAMPLING_RATE,
    )

    def save_results(df, outputs, data_file, encoder_hs_file):
        encoder_hs = outputs["encoder_hidden_states"]
        df = df.drop(columns=["raw_audio"])
        df.to_csv(f"{output_dir}/{data_file}")
        torch.save(encoder_hs, f"{output_dir}/{encoder_hs_file}")

    # training data
    save_results(
        train_df,
        train_outputs,
        train_out_file,
        f"train_enc_emb_{clean_model_name}_{clean_dataset_name}_{lang}.pt",
    )

    # test data
    save_results(
        test_df,
        test_outputs,
        f"test_{clean_model_name}_{clean_dataset_name}_{lang}.csv",
        f"test_enc_emb_{clean_model_name}_{clean_dataset_name}_{lang}.pt",
    )


if __name__ == "__main__":
    stime = time.time()
    print("RUN STARTED")
    fire.Fire(main)
    print(f"RUN COMPLETED ({time.time() - stime:.0f} SECONDS)")
