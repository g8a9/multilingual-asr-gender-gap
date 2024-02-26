from sklearn.cluster import HDBSCAN
import torch
import os
from scipy.spatial.distance import cosine
import umap
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

sns.set_theme("paper")

SPLITS = ["train", "validation", "test"]

LANG_TO_CONFIG_MAPPING = {
    "de": "de_de",
    "en": "en_us",
    "nl": "nl_nl",
    "ru": "ru_ru",
    "sr": "sr_rs",
    "it": "it_it",
    "fr": "fr_fr",
    "es": "es_419",
    "ca": "ca_es",
    "pt": "pt_br",
    "sw": "sw_ke",
    "yo": "yo_ng",
    "ja": "ja_jp",
    "hu": "hu_hu",
    "ar": "ar_eg",
    "fi": "fi_fi",
    "ro": "ro_ro",
    "cs": "cs_cz",
    "sk": "sk_sk",
}

id_to_gender = {1: "female", 0: "male"}

if __name__ == "__main__":
    speaker_counts = list()

    for lang, lang_code in tqdm(LANG_TO_CONFIG_MAPPING.items(), desc="Lang"):

        outfile = f"./results-interim-asr-performance-gap/dataset_statistics/fleurs/speaker_info_{lang}.csv"
        if os.path.exists(outfile):
            print("Output file exists alread. Skipping...")
            continue

        dataset = load_dataset(
            "google/fleurs",
            lang_code,
            num_proc=7,
            trust_remote_code=True,
        ).remove_columns("audio")

        all_embeds = list()
        genders = list()
        splits = list()

        for split in SPLITS:
            curr_data = dataset[split]
            emb_file = f"./results-interim-asr-performance-gap/embeddings/fleurs_ecapa_voxceleb_{split}_{lang}.pt"
            try:
                embeds = torch.load(emb_file)
            except Exception as e:
                print(emb_file, "not found. Skipping...")
                raise e
            all_embeds.append(embeds)
            genders.extend([r["gender"] for r in curr_data])
            splits.extend([split] * len(curr_data))

        embeds = torch.cat(all_embeds)
        genders = np.array(genders)
        speaker_ids = np.array(["unk"] * len(genders), dtype="object")
        assert genders.shape[0] == embeds.shape[0]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=120, ncols=2)

        gender_counts = {"lang": lang}
        for curr_gender in [0, 1]:  # in fleurs 1 is female, 0 is male
            curr_embeds = embeds[genders == curr_gender, :]

            print("Clustering input size:", curr_embeds.shape, lang)
            hdb = HDBSCAN(
                min_cluster_size=2, metric=cosine, n_jobs=7
            )  # assuming a speaker has at least 2 recordings, this should be safe
            labels = hdb.fit_predict(curr_embeds)

            # labels = set([l for l in labels.tolist() if l != -1])
            labels_set = set(labels)  # -1 for outliers

            print(
                f"Lang: {lang}, Gender: {id_to_gender[curr_gender]}, number of speakers: {len(labels_set) - 1}"
            )

            gender_counts[id_to_gender[curr_gender]] = len(labels_set)

            # compute umap projections
            reducer = umap.UMAP()
            curr_embeds = reducer.fit_transform(curr_embeds)

            ax[curr_gender].scatter(
                curr_embeds[:, 0],
                curr_embeds[:, 1],
                c=[sns.color_palette("hls", len(labels_set))[x] for x in labels],
            )
            ax[curr_gender].set_title(
                f"{id_to_gender[curr_gender]}, speaker count: {len(labels_set)}"
            )

            labels = [f"{id_to_gender[curr_gender]}_{l}" for l in labels]
            speaker_ids[genders == curr_gender] = labels

        speaker_counts.append(gender_counts)

        fig.suptitle(f"FLEURS, language: {lang}")
        # fig.tight_layout()
        fig.savefig(
            f"./results-interim-asr-performance-gap/charts/fleurs/umap_hdbscan_{lang}.pdf"
        )

        stats = pd.DataFrame(
            {
                "rid": np.arange(len(genders)),  # type: ignore
                "split": splits,
                "client_id": speaker_ids,
                "gender": genders,
            }
        )

        print("Saving output file to", outfile)
        stats.to_csv(
            outfile,
            index=None,
        )  # type: ignore

    # speaker_counts = pd.DataFrame(speaker_counts).set_index("lang")
    # speaker_counts.to_csv(
    #     "./results-interim-asr-performance-gap/dataset_statistics/fleurs/ecapa_voxceleb_speaker_count.csv"
    # )
