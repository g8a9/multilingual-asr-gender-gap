import json
from itertools import product
from utils import CV_LANGS, VOXPOPULI_LANGS
from mozilla_cv import LANGS_TO_LOAD_REMOTE

SPLITS = ["train", "validation", "test"]
SPLITS = ["validation", "test"]
MODELS = ["openai/whisper-large-v3", "facebook/seamless-m4t-v2-large"]
DATASETS = [
    "mozilla-foundation/common_voice_16_0",
    "google/fleurs",
    "facebook/voxpopuli",
]
OUTPUT_FILE = "../configs/co2_emission_transcription_config.json"

LONG_LANGS = {"en", "ca", "de", "es", "fr", "it"}
PROCESS_SLICES = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]

if __name__ == "__main__":
    configs = dict()
    configuration_id = 0
    for dataset, model, split in product(DATASETS, MODELS, SPLITS):
        if "voxpopuli" in dataset:
            langs = VOXPOPULI_LANGS
        else:
            langs = CV_LANGS

        langs = ["ar", "fi", "sk", "cs", "ro"]
        for lang in langs:
            # if "mozilla" in dataset:
            #     if lang not in LANGS_TO_LOAD_REMOTE:
            #         load_type = "local"
            #         local_dir = (
            #             "/data/milanlp/attanasiog/fair_asr/cv-corpus-16.0-2023-12-06"
            #         )
            #     else:
            #         load_type = "remote"
            #         local_dir = None
            # else:
            load_type = "remote"
            local_dir = None

            if "fleurs" in dataset:
                reference_col = "raw_transcription"
            elif "mozilla" in dataset:
                reference_col = "sentence"
            else:
                reference_col = "raw_text"

            if "voxpopuli" in dataset:
                speaker_id_col = "speaker_id"
            else:
                speaker_id_col = "client_id"

            if split == "train" and lang in LONG_LANGS and "mozilla" in dataset:
                for process_slice in PROCESS_SLICES:
                    configs[configuration_id] = {
                        "dataset": dataset,
                        "model": model,
                        "lang": lang,
                        "split": split,
                        "load_type": load_type,
                        "local_dir": local_dir,
                        "process_slice": process_slice,
                        "reference_col": reference_col,
                        "speaker_id_col": speaker_id_col,
                    }
                    configuration_id += 1
            else:
                configs[configuration_id] = {
                    "dataset": dataset,
                    "model": model,
                    "lang": lang,
                    "split": split,
                    "load_type": load_type,
                    "local_dir": local_dir,
                    "process_slice": None,
                    "reference_col": reference_col,
                    "speaker_id_col": speaker_id_col,
                }
                configuration_id += 1

    print(f"Generated {len(configs)} configs.")

    with open(OUTPUT_FILE, "w") as fp:
        json.dump(configs, fp, indent=2)
