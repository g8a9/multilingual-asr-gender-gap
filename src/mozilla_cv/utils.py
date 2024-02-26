import joblib
import pandas as pd
import librosa
import csv


class MozillaCVDataset:
    def __init__(
        self, local_dir: str, lang: str, split: str, decode_audio: bool = True
    ) -> None:
        self.lang = lang
        self.split = split

        def load_split(s):
            split_file = (
                f"{local_dir}/{lang}/{s}.tsv"
                if s != "validation"
                else f"{local_dir}/{lang}/dev.tsv"
            )

            df = pd.read_csv(
                split_file, sep="\t", quoting=csv.QUOTE_NONE, encoding="utf-8"
            ).rename(columns={"path": "audio"})

            if decode_audio:
                df["audio"] = df["audio"].apply(
                    lambda x: f"{local_dir}/{lang}/clips/{x}"
                )

            df["split"] = s
            return df

        if split == "all":
            self.data = pd.concat(map(load_split, ["train", "validation", "test"]))
        else:
            self.data = load_split(split)

    def validate_audio(self, n_jobs: int = 4, sr=16000):
        def is_valid_audio(path):
            try:
                arr, rs = librosa.load(path, sr=sr)
                return True
            except Exception as e:
                return False

        is_valid = joblib.Parallel(n_jobs=n_jobs, verbose=5, batch_size=1000)(
            joblib.delayed(is_valid_audio)(path) for path in self.data["audio"]
        )

        self.data["is_valid"] = is_valid
        # self.data["array"] = [x[1] for x in is_valid]
