import torch
import numpy as np
from sklearn.cluster import HDBSCAN
from scipy.spatial.distance import cosine
from typing import List
from .extract_unique_users import LANG_TO_CONFIG_MAPPING


ID_2_GENDER_MAP = {1: "female", 0: "male"}


def extract_fleurs_speaker_ids(lang, mask: List[int], split: str = "all"):
    if split == "all":
        all_embeds = list()
        for s in ["train", "validation", "test"]:
            outfile = f"/data/milanlp/attanasiog/fair_asr/embeddings/fleurs_ecapa_voxceleb_{s}_{lang}.pt"
            embeds = torch.load(outfile)
            all_embeds.append(embeds)
        embeds = torch.cat(all_embeds)
    else:
        outfile = f"/data/milanlp/attanasiog/fair_asr/embeddings/fleurs_ecapa_voxceleb_{split}_{lang}.pt"
        embeds = torch.load(outfile)

    embeds = embeds[mask, :]

    print("Clustering input size:", embeds.shape)
    hdb = HDBSCAN(min_cluster_size=2, metric=cosine)
    labels = hdb.fit_predict(embeds)
    print("Number of speakers:", len(np.unique(labels)))
    return labels
