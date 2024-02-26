import gc
import csv
import json
import logging
import math
import os
import time

import fire
import librosa
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from datasets import Dataset, load_dataset
from tqdm import tqdm
from fleurs import LANG_TO_CONFIG_MAPPING
from datasets import Audio
from codecarbon import track_emissions

from transcriber import SimpleTranscriber

logger = logging.getLogger(__name__)


TARGET_SAMPLING_RATE = 16_000
MAX_LENGTH_SECONDS = 30


@track_emissions
def main(
    config_file: str,
    config_id: int,
    output_dir: str,
    dry_run: bool = False,
    batch_size: int = 1,
    num_workers: int = 1,
    overwrite_output: bool = False,
    enable_chunk_decoding: bool = False,  # set to True to prevent OOM
    chunk_size: int = 3000,
    # column names to change depending on the dataset used
    reference_col: str = "sentence",
    speaker_id_col: str = "client_id",
    gender_col: str = "gender",
):
    with open(config_file) as fp:
        configs = json.load(fp)

    config = configs[str(config_id)]
    print("Loaded config:")
    print(config)

    lang = config["lang"]
    dataset = config["dataset"]
    split = config["split"]
    model = config["model"]

    clean_model_name = model.replace("/", "--")
    clean_dataset_name = dataset.replace("/", "--")
    load_type = config["load_type"]
    local_dir = config.get("local_dir", None)

    reference_col = config.get("reference_col", reference_col)
    speaker_id_col = config.get("speaker_id_col", speaker_id_col)

    process_slice = config.get("process_slice", None)
    if process_slice:
        print(f"### Slicing the dataset wiht slice: {process_slice}")

    out_file = (
        (f"{output_dir}/{clean_model_name}_{clean_dataset_name}_{split}_{lang}.tsv")
        if not process_slice
        else (
            f"{output_dir}/{clean_model_name}_{clean_dataset_name}_{split}_{lang}_{process_slice}.tsv"
        )
    )
    if os.path.exists(out_file) and not overwrite_output:
        print(f"Output file {out_file} exists already. Skipping...")
        return

    lang_code = lang if "fleurs" not in dataset else LANG_TO_CONFIG_MAPPING[lang]

    if load_type == "remote":
        print("Loading remote dataset.", dataset, lang_code, split, num_workers)
        data = load_dataset(
            dataset,
            lang_code,
            split=split,
            num_proc=num_workers,
            trust_remote_code=True,
        )

        if "mozilla" in dataset:
            data = data.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))

    elif load_type == "local":
        print("Starting to load and decode local audio files.")
        split_file = (
            f"{local_dir}/{lang}/{split}.tsv"
            if split != "validation"
            else f"{local_dir}/{lang}/dev.tsv"
        )
        df = pd.read_csv(
            split_file, sep="\t", quoting=csv.QUOTE_NONE, encoding="utf-8"
        ).rename(columns={"path": "audio"})
        df["audio"] = df["audio"].apply(lambda x: f"{local_dir}/{lang}/clips/{x}")
        data = Dataset.from_pandas(df)

        print("Loading finished.")
    else:
        raise ValueError("Load type unknown")

    print("Initial len", len(data))

    if dry_run:
        print("Running dry run (first 128 samples)...")
        data = data.select(range(128))

    if process_slice:
        left_p, right_p = process_slice.split("-")
        left_idx = int(float(left_p) * len(data))
        right_idx = int(float(right_p) * len(data))
        print(
            f"Processing only data between {left_p} (idx: {left_idx}) and {right_p} (idx: {right_idx})"
        )
        data = data.select(range(left_idx, right_idx))  # type: ignore

    print("Final len", len(data))  # type: ignore
    data = data.add_column("rid", [f"{split}_{i}" for i in range(len(data))])  # type: ignore

    #####
    # Prepare the pipeline and model
    #####
    transcriber = SimpleTranscriber(
        model_name_or_path=model,
        tgt_lang=lang,
        torch_dtype=torch.bfloat16,
        # chunk_length_s=30,
        device="cuda",
    )
    print("Transcriber loaded")

    def load_and_resample_to_16khz(examples):
        resampled = list()
        is_valid = list()
        lengths = list()
        for path in examples["audio"]:
            try:
                arr, _ = librosa.load(path, sr=TARGET_SAMPLING_RATE)
                resampled.append(arr)
                is_valid.append(True)
                lengths.append(arr.shape[0])
            except Exception as e:
                print("Error while decoding and/or resampling. Skipping file:", path)
                print(e)
                resampled.append(np.zeros((1,), dtype=np.float32))
                is_valid.append(False)
                lengths.append(-1)
        return {"audio": resampled, "is_valid": is_valid, "length": lengths}

    def transcribe_chunk(data, transcriber, load_and_resample: bool):
        """
        CV data is in 48 kHz, VP and FLEURS are in 16 kHz.
        Whisper and Seamless require 16 kHZ sampled data.
        """

        if load_and_resample:
            # 1. decode and resample to the target sampling rate
            stime = time.time()
            data = data.map(
                load_and_resample_to_16khz,
                batched=True,
                num_proc=num_workers,
                desc="Loading and resampling to 16khz",
            )
            # 2. filter out the invalid records
            data = data.filter(
                lambda examples: examples["is_valid"],
                batched=True,
                num_proc=num_workers,
                desc="Filter valid records",
            )

            print(f"Time to load, resample, and filter:", time.time() - stime)

            raw_audio = data["audio"]
            chunk_max_length = max(data["length"])
        else:
            raw_audio = [d["array"] for d in data["audio"]]
            chunk_max_length = max([a.shape[0] for a in raw_audio])

        # 2. get the transcriptions
        print("Max length (s) found in chunk:", chunk_max_length / TARGET_SAMPLING_RATE)
        print(f"Trimming to: {MAX_LENGTH_SECONDS}")
        if chunk_max_length / TARGET_SAMPLING_RATE > MAX_LENGTH_SECONDS:
            print("SOME AUDIO WILL GET TRIMMED")

        print(f"Transcribing chuck of {len(raw_audio)} samples.")
        print("Some references", data[reference_col][:3])

        transcriptions = transcriber(
            raw_audio=raw_audio,
            sampling_rate=TARGET_SAMPLING_RATE,
            batch_size=batch_size,
            num_workers=4,
            max_length=MAX_LENGTH_SECONDS * TARGET_SAMPLING_RATE,
        )

        try:
            # FLEURS has no speaker id
            speaker_ids = data[speaker_id_col]
        except:
            speaker_ids = [None] * len(data)

        return {
            "reference": data[reference_col],
            "transcription": transcriptions,
            "client_id": speaker_ids,
            "gender": data[gender_col],
            "rid": data["rid"],
        }

    # split decoding + transcribing in chunks
    if enable_chunk_decoding:
        idxs = np.arange(len(data))
        n_chunks = math.ceil(len(data) / chunk_size)
        batches = np.array_split(idxs, n_chunks)

        transcriptions = list()
        references = list()
        clients = list()
        genders = list()
        rids = list()
        for batch in tqdm(batches, desc="Chunk", total=n_chunks):
            curr_data = data.select(batch)
            # curr_data = curr_data.cast_column("audio", Audio()) # loading demanded to HF
            r = transcribe_chunk(
                curr_data,
                transcriber,
                load_and_resample=(load_type == "local" and "mozilla" in dataset),
            )

            print("Verify")
            print(r["transcription"][:3])
            print(r["reference"][:3])

            transcriptions.extend(r["transcription"])
            references.extend(r["reference"])
            clients.extend(r["client_id"])
            genders.extend(r["gender"])
            rids.extend(r["rid"])
            gc.collect()
    else:
        r = transcribe_chunk(data, transcriber)
        transcriptions = r["transcription"]
        references = r["reference"]
        clients = r["client_id"]
        genders = r["gender"]
        rids = r["rid"]

    result = {
        "rid": rids,
        "client_id": clients,
        "gender": genders,
        "transcription": transcriptions,
        "reference": references,
    }

    results = pd.DataFrame(result).set_index("rid")
    results.to_csv(out_file, encoding="utf-8", sep="\t")


if __name__ == "__main__":
    stime = time.time()
    print("RUN STARTED")
    fire.Fire(main)
    print(f"RUN COMPLETED ({time.time() - stime:.0f} SECONDS)")
