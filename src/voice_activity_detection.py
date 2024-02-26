import torch
from datasets import load_dataset, concatenate_datasets
import fire
import time
from fleurs import LANG_TO_CONFIG_MAPPING
from pyannote.audio import Pipeline
from tqdm import tqdm
import pandas as pd
from mozilla_cv import MozillaCVDataset, LANGS_TO_LOAD_REMOTE
import librosa
import os


def main(
    output_dir: str,
    dataset_name: str,
    lang: str,
    split: str = "all",
    num_workers: int = 4,
    overwrite_output: bool = False,
):
    load_local_CV = "mozilla" in dataset_name and lang not in LANGS_TO_LOAD_REMOTE

    dataset_id = dataset_name.replace("/", "--")
    lang_code = LANG_TO_CONFIG_MAPPING[lang] if "fleurs" in dataset_name else lang
    output_file = f"support_{dataset_id}_{split}_{lang}.csv"
    if os.path.exists(f"{output_dir}/{output_file}") and not overwrite_output:
        print(f"Output file {output_file} exists already. Skipping...")
        return

    if load_local_CV:
        dataset = MozillaCVDataset(
            "/data/milanlp/attanasiog/fair_asr/cv-corpus-16.0-2023-12-06",
            lang,
            split,
            decode_audio=True,
        )
        dataset.validate_audio(n_jobs=num_workers)
        print("Validity check:", dataset.data["is_valid"].value_counts())
        dataset = dataset.data.loc[dataset.data["is_valid"] == True]
    else:
        if split == "all":
            splits = list()
            for curr_split in ["train", "validation", "test"]:
                d = load_dataset(
                    dataset_name, lang_code, split=curr_split, num_proc=num_workers
                )
                print("Loaded split:", curr_split, lang_code, len(d))
                d = d.add_column("split", [curr_split] * len(d))  # type: ignore
                splits.append(d)

            dataset = concatenate_datasets(splits)
        else:
            dataset = load_dataset(dataset_name, lang_code, split=split)

    print("Loaded dataset:", dataset_name, split, len(dataset))

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection").to(
        torch.device("cuda:0")
    )

    # dataset = dataset.select([0, 100, 1000, 1765, 1773, 1777])
    durations = list()
    supports = list()
    iterator = dataset.iterrows() if load_local_CV else dataset
    for row in tqdm(iterator, desc="Record", total=len(dataset)):
        # print(row["audio"])

        if load_local_CV:
            idx, row = row[0], row[1]
            sample_rate = 48000
            waveform, sr = librosa.load(row["audio"], sr=48000)
            waveform = torch.tensor(waveform.reshape(1, -1)).to(
                dtype=torch.float32, device=pipeline.device
            )
        else:
            waveform = torch.tensor(row["audio"]["array"].reshape(1, -1)).to(
                dtype=torch.float32, device=pipeline.device
            )
            # print(waveform.dtype)
            sample_rate = row["audio"]["sampling_rate"]
            # print(waveform.shape[0], waveform.shape[1], len(waveform.shape))

        speech = pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }
        )
        durations.append(speech.get_timeline().support().duration())
        supports.append(len(speech.get_timeline().support()))
        # print(pipeline(row["audio"]))

        # print("SPEECH", speech)
        # print("TIMELINE", speech.get_timeline())
        # print("LEN TIMELINE", len(speech.get_timeline()))
        # print("SUPPORT", speech.get_timeline().support())
        # print("DURATION", speech.get_timeline().support().duration())
        # print("LEN SUPPORT", len(speech.get_timeline().support()))

    df = pd.DataFrame(
        {
            "rid": list(range(len(dataset))),
            # "id": dataset["id"],
            "split": dataset["split"],
            "gender": dataset["gender"],
            "duration": durations,
            "support": supports,
        }
    )
    df.to_csv(f"{output_dir}/{output_file}", index=False)
    # print(dataset[1765]["audio"]["array"])


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    print(f"Total time: {time.time() - stime:.2f}s")
