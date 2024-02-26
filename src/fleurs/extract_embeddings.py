import os
import fire
from datasets import load_dataset
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm
import torch
import time
from extract_unique_users import LANG_TO_CONFIG_MAPPING


def main(lang):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/data/milanlp/attanasiog/fair_asr/models",
        run_opts={"device": "cuda"},
    )

    dataset = load_dataset(
        "google/fleurs", LANG_TO_CONFIG_MAPPING[lang], streaming=True
    )

    for split in ["test"]:  # ["train", "validation", "test"]:
        curr_data = dataset[split]
        outfile = f"./results-interim-asr-performance-gap/embeddings/fleurs_ecapa_voxceleb_{split}_{lang}.pt"

        # if os.path.exists(outfile):
        #     print("Output file exists alread. Skipping...")
        #     return
        # print(curr_data)
        # loader = DataLoader(KeyDataset(curr_data, "audio"), batch_size=16, shuffle=False, pin_memory=True)

        # curr_batch = list()
        embeddings = list()
        for i, row in tqdm(
            enumerate(curr_data), desc=f"Processing {split}", leave=False
        ):
            raw_audio = row["audio"]["array"]

            if row["audio"]["sampling_rate"] != 16000:
                raw_audio = torchaudio.functional.resample(
                    raw_audio, row["audio"]["sampling_rate"], 16_000
                )

            raw_audio = torch.from_numpy(raw_audio)
            embed = classifier.encode_batch(raw_audio).squeeze(
                0
            )  # --> (1, hidden dim, which should be 192)
            embeddings.append(embed.to(device="cpu", dtype=torch.float32))

        torch.save(torch.cat(embeddings), outfile)


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    print(f"Elapsed {time.time() - stime:.0f} seconds.")
