import pandas as pd

MODELS = ["facebook/seamless-m4t-v2-large"]
MODELS = ["openai/whisper-large-v3", "facebook/seamless-m4t-v2-large"]
DATASETS = [
    "mozilla-foundation/common_voice_16_0",
    "google/fleurs",
    "facebook/voxpopuli",
]

LONG_LANGS = {"en", "ca", "de", "es", "fr", "it"}
PROCESS_SLICES = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]
TRANSCRIPTIONS_DIR = "../results-interim-asr-performance-gap/bs4_new/transcriptions"

dataset_id = DATASETS[0].replace("/", "--")
for model in MODELS:
    model_id = model.replace("/", "--")
    for lang in LONG_LANGS:
        slices = list()
        for suffix in PROCESS_SLICES:
            df = pd.read_csv(
                f"{TRANSCRIPTIONS_DIR}/{model_id}_{dataset_id}_train_{lang}_{suffix}.tsv",
                encoding="utf-8",
                sep="\t",
            )
            df = df.drop(columns=["rid"])
            slices.append(df)
        df = pd.concat(slices)
        df["rid"] = [f"train_{i}" for i in range(len(df))]
        df = df.set_index("rid")
        df.to_csv(
            f"{TRANSCRIPTIONS_DIR}/{model_id}_{dataset_id}_train_{lang}.tsv",
            sep="\t",
            encoding="utf-8",
        )
