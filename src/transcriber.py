from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

TARGET_SAMPLING_RATE = 16_000

WHISPER_CODE_TO_LANG = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# see https://huggingface.co/facebook/seamless-m4t-v2-large#supported-languages
SEAMLESS_CODE_TO_LANG = {
    "de": "deu",
    "en": "eng",
    "nl": "nld",
    "ru": "rus",
    "sr": "srp",
    "it": "ita",
    "fr": "fra",
    "es": "spa",
    "ca": "cat",
    "pt": "por",
    "da": "dan",
    "sw": "swh",
    "yo": "yor",
    "hi": "hin",
    "ja": "jpn",
    "hu": "hun",
    "ar": "arb",
    "fi": "fin",
    "ro": "ron",
    "hr": "hrv",
    "cs": "ces",
    "sk": "slk",
}

# def _build_loader(raw_audio, batch_size, num_workers, max_length):
#     class SimpleDataset(torch.utils.data.Dataset):
#         def __init__(self, array_list):
#             self.array_list = array_list

#         def __getitem__(self, index):
#             return self.array_list[index]

#         def __len__(self):
#             return len(self.array_list)

#     def collate_pad_and_trim(batch):
#         """
#         Pad/trim all audios to a max length. Then, create a batch.
#         """
#         # stime = time.time()
#         # 1. Find the longest item in the batch
#         lengths = [len(e) for e in batch]
#         batch_max_length = max(lengths)

#         if batch_size > 1:
#             # 2. Pad every sample in the batch to min(max_length, batch_max_length)
#             target_max_length = (
#                 min(batch_max_length, max_length) if max_length else batch_max_length
#             )
#             arrays = list()
#             for array in batch:
#                 # requires padding
#                 if len(array) < target_max_length:
#                     out = np.pad(array, (0, target_max_length - len(array)))
#                 # requires trimming
#                 else:
#                     out = np.array(array[:target_max_length])

#                 arrays.append(out)

#             batch = np.stack(arrays, axis=0)

#         else:
#             batch = np.array(batch)

#         # print(f"Batch creation time:", time.time() - stime)
#         return {"audio": batch}

#     loader = torch.utils.data.DataLoader(
#         SimpleDataset(raw_audio),
#         batch_size=batch_size,
#         num_workers=num_workers,
#         collate_fn=collate_pad_and_trim,
#         pin_memory=True,
#     )

#     return loader


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, array_list):
        self.array_list = array_list

    def __getitem__(self, index):
        return self.array_list[index]

    def __len__(self):
        return len(self.array_list)


class SimpleTranscriber:
    def __init__(
        self,
        model_name_or_path: str,
        tgt_lang: str,
        device: str = "cuda",
        **init_kwargs,
    ):
        self.tgt_lang = tgt_lang
        self.model_name_or_path = model_name_or_path
        self.device = device

        # # Load Processor
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Load Model
        if "whisper" in model_name_or_path or "seamless" in model_name_or_path:
            model_cls = AutoModelForSpeechSeq2Seq
        else:
            model_cls = AutoModelForCTC

        print(f"Loading {model_name_or_path} and moving it to {device}...")
        self.model = (
            model_cls.from_pretrained(model_name_or_path, **init_kwargs)
            .to(device)
            .eval()
        )

        print("Transcriber loaded for language:", tgt_lang)

    def _build_loader(
        self,
        raw_audio: List[Union[np.ndarray, List[float]]],
        sampling_rate: int,
        batch_size: int,
        num_workers: int,
        max_length: int,
    ):
        pargs = dict(
            return_tensors="pt",
            sampling_rate=sampling_rate,
            return_attention_mask=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_length,
        )

        def collate_pad_and_trim(batch: List[Union[np.ndarray, List[float]]]):
            """
            Pad/trim all audios to a max length. Then, create a batch.
            """
            if "whisper" in self.model_name_or_path:
                pargs["audio"] = batch  # type: ignore
                pargs["do_normalize"] = True
                pargs["padding"] = "max_length"
            else:
                pargs["audios"] = batch  # type: ignore
                pargs["padding"] = "longest"

            inputs = self.processor(**pargs)
            return inputs

        loader = torch.utils.data.DataLoader(
            SimpleDataset(raw_audio),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_pad_and_trim,
            pin_memory=True,
        )

        return loader

    @torch.inference_mode()
    def __call__(
        self,
        raw_audio: List[Union[np.ndarray, List[float]]],
        sampling_rate: int,
        batch_size: int = 1,
        num_workers: int = 1,
        show_progress_bar: bool = True,
        max_length: int = 480000,
        **generation_kwargs,
    ):
        """
        Transcribe a list of audio samples.

        Args:
            raw_audio (List[Union[np.ndarray, List[float]]]): List of raw audio data.
            sampling_rate (int): Sampling rate of the audio data.
            batch_size (int, optional): Number of audio samples per batch. Defaults to 1.
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 1.
            show_progress_bar (bool, optional): Whether to show a progress bar during inference. Defaults to True.
            max_length (int, optional): Maximum length of audio samples in the batch. Defaults to 480000.
            **generation_kwargs: Additional keyword arguments for the generation process.

        Returns:
            List[str]: List of transcriptions for each audio sample.
        """

        loader = self._build_loader(
            raw_audio=raw_audio,
            sampling_rate=sampling_rate,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
        )

        transcriptions = list()
        for idx, batch in tqdm(
            enumerate(loader),
            desc="Batch",
            disable=not show_progress_bar,
            total=len(loader),
            # miniters=int(len(loader) / 100),
        ):
            batch = {
                k: v.to(dtype=self.model.dtype, device=self.model.device)
                for k, v in batch.items()
            }

            if "whisper" in self.model_name_or_path:
                # inputs = inputs.input_features
                generation_kwargs["forced_decoder_ids"] = (
                    self.processor.get_decoder_prompt_ids(
                        language=WHISPER_CODE_TO_LANG[self.tgt_lang], task="transcribe"
                    )
                )

                predicted_ids = self.model.generate(**batch, **generation_kwargs)

            elif "seamless" in self.model_name_or_path:
                generation_kwargs |= {
                    "tgt_lang": SEAMLESS_CODE_TO_LANG[self.tgt_lang],
                }

                predicted_ids = self.model.generate(
                    **batch, **generation_kwargs
                )  # [0].cpu().detach()
            else:  # A CTC model
                logits = self.model(**batch, **generation_kwargs).logits
                predicted_ids = logits.argmax(-1)

            results = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            transcriptions.extend(results)

        return transcriptions

    def forward_encoder(
        self,
        raw_audio: List[Union[np.ndarray, List[float]]],
        sampling_rate: int,
        batch_size: int = 1,
        num_workers: int = 1,
        show_progress_bar: bool = False,
        max_length: int = 480000,
        **forward_kwargs,
    ):
        output_hidden_states = forward_kwargs.get("output_hidden_states", False)

        loader = self._build_loader(
            raw_audio=raw_audio,
            sampling_rate=sampling_rate,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
        )

        hidden_states_list = list()

        for batch in tqdm(loader, desc="Batch", disable=not show_progress_bar):

            batch = batch.to(dtype=self.model.dtype, device=self.device)

            out = self.model.model.encoder(**batch, **forward_kwargs)

            if output_hidden_states:
                # hs = torch.stack(out.hidden_states) # (num_layers, bs, seq_len, hs)

                # print("LEN HIDDEN STATES", len(out.hidden_states))
                last_hs = out.hidden_states[-1]  # (bs, seq_len, hsize)
                # print("SHAPE LAST HS", last_hs.shape)
                # print("SHAPE FIRST HS", out.hidden_states[0].shape)
                # print("SHAPE SECOND TO LAST HS", out.hidden_states[-2].shape)
                # hs = hs.transpose(0, 1) # (bs, num_layers, seq_len, hs)
                hidden_states_list.append(last_hs.cpu().detach())

        output = dict()

        if output_hidden_states:
            # (dataset, num_layers, seq_len, hs)
            output["encoder_hidden_states"] = torch.cat(hidden_states_list)

        return output


class SimpleTranscriberPipeline:
    def __init__(self, model_name_or_path: str, lang: str, **pipeline_kwargs):
        self.lang = lang
        self.model_name_or_path = model_name_or_path

        # https://huggingface.co/GeoffVdr/whisper-medium-nlcv11/discussions/1
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_name_or_path,
            **pipeline_kwargs,
        )

        if "whisper" in model_name_or_path:
            self.pipe.model.config.forced_decoder_ids = (
                self.pipe.tokenizer.get_decoder_prompt_ids(
                    language=lang, task="transcribe"
                )
            )

    def __call__(
        self,
        raw_audio,
        batch_size: int = 1,
        show_progress_bar: bool = True,
        **generation_kwargs,
    ):
        def iterate_data(dataset):
            for _, item in enumerate(dataset):
                yield np.array(item)

        transcriptions = list()

        for out in tqdm(
            self.pipe(
                iterate_data(raw_audio), batch_size=batch_size, **generation_kwargs
            ),
            desc="Batch",
            total=len(raw_audio),
            disable=(not show_progress_bar),
        ):
            transcriptions.append(out["text"])

        return transcriptions
