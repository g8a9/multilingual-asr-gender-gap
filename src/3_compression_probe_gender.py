import fire
import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import trange
import numpy as np
from typing import List
from transformers import set_seed

GENDER_2_ID = {"female": 1, "male": 0, "other": 1}
FRACTIONS = [0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100]
FRACTIONS_OTHER = [25, 35, 50, 60, 65, 80, 100]


class MDLProbe:
    def __init__(self, fractions=FRACTIONS, num_workers: int = 4):
        self.fractions = [f / 100 for f in fractions]
        self.num_workers = num_workers

    def _step(self, X_train, y_train, X_test, y_test):
        logreg = LogisticRegression(
            n_jobs=self.num_workers,
            random_state=42,
            class_weight="balanced",
            max_iter=1000,
        )
        logreg.fit(X_train, y_train)
        proba = logreg.predict_proba(X_test)
        proba = proba[:, y_test]  # probs for the true class
        loss = np.sum(-np.log2(proba))
        return loss

    def evaluate(
        self,
        X: np.ndarray,
        y: List[str],
    ):
        losses = list()
        for i in range(
            len(self.fractions) - 1
        ):  # skip the last step (100%) as it does not have a test set
            fr_X_train, fr_y_train = (
                X[: int(len(X) * self.fractions[i]), :],
                y[: int(len(y) * self.fractions[i])],
            )
            fr_X_test, fr_y_test = (
                X[
                    int(len(X) * self.fractions[i]) : int(
                        len(X) * self.fractions[i + 1]
                    ),
                    :,
                ],
                y[
                    int(len(y) * self.fractions[i]) : int(
                        len(X) * self.fractions[i + 1]
                    )
                ],
            )

            # if test set contains only one class, use as loss the average loss so far
            # print(f"Train: {fr_X_train.shape}, Test: {fr_X_test.shape}")
            # print(fr_y_train)
            # print(fr_y_train, fr_y_test)
            if len(np.unique(fr_y_train)) == 1:
                if i == 0:  # the first step is too small and we can't use it
                    continue
                loss = np.nanmean(losses)
            else:
                loss = self._step(fr_X_train, fr_y_train, fr_X_test, fr_y_test)

            losses.append(loss)

        # online codelength
        num_classes = len(np.unique(y))
        first_train_size = int(X.shape[0] * self.fractions[0])
        online_codelength = first_train_size * np.log2(num_classes) + np.sum(losses)
        uniform_codelength = X.shape[0] * np.log2(num_classes)
        compression = round(uniform_codelength / online_codelength, 2)
        return {
            "online_codelength": online_codelength,
            "uniform_codelength": uniform_codelength,
            "compression": compression,
        }


class LogRegProbe:
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers

    def evaluate(self, X_train, y_train, X_test, y_test):
        logreg = LogisticRegression(
            n_jobs=self.num_workers,
            random_state=42,
            class_weight="balanced",
            max_iter=1000,
        )
        logreg.fit(X_train, y_train)

        y_pred = logreg.predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        return {
            "f1_macro": f1_macro,
            "acc": acc,
        }


def main(
    model,
    dataset,
    lang,
    output_dir: str,
    embedding_dir: str,
    num_workers: int = 1,
    probe_type="logreg",
    frame_step: int = 1,
    random_shift_labels: bool = False,
    seed: int = 42,
):
    set_seed(seed)
    model_id = model.replace("/", "--")
    dataset_id = dataset.replace("/", "--")

    output_file = (
        f"results_{model_id}_{dataset_id}_{lang}_{probe_type}.csv"
        if not random_shift_labels
        else f"results_{model_id}_{dataset_id}_{lang}_{probe_type}_random_shift.csv"
    )

    # These are splits we created in advance with no overalapping speakers
    train_df = pd.read_csv(
        os.path.join(embedding_dir, f"train_{model_id}_{dataset_id}_{lang}.csv")
    )
    test_df = pd.read_csv(
        os.path.join(embedding_dir, f"test_{model_id}_{dataset_id}_{lang}.csv")
    )
    train_embs = torch.load(
        os.path.join(embedding_dir, f"train_enc_emb_{model_id}_{dataset_id}_{lang}.pt")
    )
    test_embs = torch.load(
        os.path.join(embedding_dir, f"test_enc_emb_{model_id}_{dataset_id}_{lang}.pt")
    )

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Train embs: {train_embs.shape}, Test embs: {test_embs.shape}")

    TARGET_INDEXES = 1000
    results = list()
    print(f"Probing {TARGET_INDEXES} positions with {probe_type} and step {frame_step}")

    # if probe_type == "logreg":
    # let's get the first embedding and train a logistic regression on top of it
    y_train = [GENDER_2_ID[i] for i in train_df["gender"].tolist()]
    y_test = [GENDER_2_ID[i] for i in test_df["gender"].tolist()]

    if random_shift_labels:
        # we randomly shift the labels to make the task harder
        print("Randomly shifting labels")
        np.random.shuffle(y_train)

    # fix a permutation of mdl shuffle of the dataset
    if probe_type == "mdl":
        mdl_permutation_idx = np.random.permutation(len(train_embs) + len(test_embs))

    for target_emb_idx in trange(0, TARGET_INDEXES, frame_step, desc="Pos"):
        X_train = train_embs[:, target_emb_idx, :].to(dtype=torch.float16).numpy()
        X_test = test_embs[:, target_emb_idx, :].to(dtype=torch.float16).numpy()

        if probe_type == "logreg":
            probe = LogRegProbe(num_workers=num_workers)
            res = probe.evaluate(X_train, y_train, X_test, y_test)
        elif probe_type == "mdl":
            # we concatenate train and test to get the full dataset
            X = np.concatenate((X_train, X_test), axis=0)
            y = y_train + y_test
            # shuffle X and y with the permutation defined previously
            X = X[mdl_permutation_idx]
            y = [y[i] for i in mdl_permutation_idx]

            probe = MDLProbe(
                num_workers=num_workers,
                fractions=FRACTIONS_OTHER if "other" in embedding_dir else FRACTIONS,
            )
            res = probe.evaluate(X, y)
        else:
            raise ValueError(f"Unknown probe type {probe_type}")

        result_df = {
            "model": model,
            "dataset": dataset,
            "lang": lang,
            "target_emb_idx": target_emb_idx,
            **res,
        }
        results.append(result_df)

    results = pd.DataFrame(results)
    results.to_csv(
        os.path.join(output_dir, output_file),
        index=None,
    )  # type: ignore


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    print(f"DONE, ELAPSED {time.time() - stime:.0f} seconds")
