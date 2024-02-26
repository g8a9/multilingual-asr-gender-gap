import contextlib
import json
import math
import os
import time
import wave
from collections import namedtuple
from typing import Union

import fire
import joblib
import librosa
import numpy as np
import pandas as pd
import parselmouth as pm
import soundfile as sf
import syllables
import webrtcvad
from datasets import concatenate_datasets, load_dataset

from fleurs import LANG_TO_CONFIG_MAPPING
from mozilla_cv import LANGS_TO_LOAD_REMOTE, MozillaCVDataset


def read_wave(path):
    """
    Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


class Frame(object):
    """
    Represents a "frame" of audio data.
    """

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def extract_audio_segment(audio_file, offset, duration, sample_rate):
    offset = int(float(offset) * sample_rate)
    n_frames = int(float(duration) * sample_rate)
    waveform, sample_rate = sf.read(
        audio_file, dtype="float32", always_2d=False, frames=n_frames, start=offset
    )
    return waveform


def vad_collector(
    sample_rate, sound_duration, frame_duration_ms, vad, frames, min_silence
):
    """
    Create intervals of silence and speech based on the vad model
    """
    # get silence intervals
    silence_intervals = []
    previous_silence_frame = None
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not is_speech:  # silence frame
            if previous_silence_frame is not None:  # there was a previous silence frame
                if (
                    frame.timestamp - previous_silence_frame
                ) > min_silence:  # silence interval longer than min
                    if (
                        len(silence_intervals) > 0
                        and silence_intervals[-1][0] == previous_silence_frame
                    ):
                        silence_intervals[-1][1] = (
                            frame.timestamp + frame_duration_ms / 1000
                        )  # update silence interval
                    else:
                        silence_intervals.append(
                            [
                                previous_silence_frame,
                                frame.timestamp + frame_duration_ms / 1000,
                            ]
                        )  # add silence interval
            else:
                previous_silence_frame = frame.timestamp
        else:
            previous_silence_frame = None

    # get speech intervals
    non_silence_intervals = []
    if len(silence_intervals) > 0:
        if silence_intervals[0][0] > 0.0:
            non_silence_intervals.append([0.0, silence_intervals[0][0]])
        for i in range(len(silence_intervals) - 1):
            current_silence_end = silence_intervals[i][1]
            next_silence_start = silence_intervals[i + 1][0]
            non_silence_intervals.append([current_silence_end, next_silence_start])
        if silence_intervals[-1][1] < sound_duration:
            non_silence_intervals.append([silence_intervals[-1][1], sound_duration])
    else:
        non_silence_intervals.append([0.0, sound_duration])

    return silence_intervals, non_silence_intervals


def _get_articulatory_duration(total_duration, silence_intervals):
    silence_duration = 0
    for interval in silence_intervals:
        silence_duration += interval[1] - interval[0]
    return total_duration - silence_duration


def _count_syllables(transcript):
    total_count = 0
    for word in transcript.split(" "):
        total_count += syllables.estimate(word)
    return total_count


def get_speaking_rate(transcript, duration):
    n_syllables = _count_syllables(transcript)
    return n_syllables / duration


def get_articulatory_rate(transcript, duration, silence_intervals):
    duration_no_silence = _get_articulatory_duration(duration, silence_intervals)
    n_syllables = _count_syllables(transcript)
    return n_syllables / duration_no_silence


def get_snr(sound, silence_intervals, non_silence_intervals):
    if len(silence_intervals) > 0:
        total_msr_noise = 0
        for interval in silence_intervals:
            msr_noise = pm.praat.call(
                sound, "Get root-mean-square", interval[0], interval[1]
            )
            total_msr_noise += msr_noise
        average_msr_noise = total_msr_noise / len(silence_intervals)

        total_msr_speech = 0
        for interval in non_silence_intervals:
            msr_speech = pm.praat.call(
                sound, "Get root-mean-square", interval[0], interval[1]
            )
            total_msr_speech += msr_speech
        average_msr_speech = total_msr_speech / len(non_silence_intervals)

        return (
            20 * math.log10(average_msr_speech / average_msr_noise)
            if average_msr_noise != 0
            else None
        )

    else:
        return None


def get_min_intensity(intensity, non_silence_intervals):
    min_intensities = []
    for interval in non_silence_intervals:
        min_intensity = pm.praat.call(
            intensity, "Get minimum", interval[0], interval[1], "Parabolic"
        )
        min_intensities.append(min_intensity)
    return sum(min_intensities) / len(min_intensities)


def get_max_intensity(intensity, non_silence_intervals):
    max_intensities = []
    for interval in non_silence_intervals:
        max_intensity = pm.praat.call(
            intensity, "Get maximum", interval[0], interval[1], "Parabolic"
        )
        max_intensities.append(max_intensity)
    return sum(max_intensities) / len(max_intensities)


def get_mean_intensity(intensity, non_silence_intervals):
    mean_intensities = []
    for interval in non_silence_intervals:
        mean_intensity = pm.praat.call(
            intensity, "Get mean", interval[0], interval[1], "energy"
        )
        mean_intensities.append(mean_intensity)
    return sum(mean_intensities) / len(mean_intensities)


def get_sd_intensity(intensity, non_silence_intervals):
    sd_intensities = []
    for interval in non_silence_intervals:
        sd_intensity = pm.praat.call(
            intensity,
            "Get standard deviation",
            interval[0],
            interval[1],
        )
        sd_intensities.append(sd_intensity)
    return sum(sd_intensities) / len(sd_intensities)


# TARGET_SAMPLING_RATE = 16_000

AudioFeatures = namedtuple(
    "AudioFeatures",
    [
        "speaking_rate",
        "articulatory_rate",
        "snr",
        "min_intensity",
        "max_intensity",
        "mean_intensity",
        "sd_intensity",
        "min_pitch",
        "max_pitch",
        "mean_pitch",
        "sd_pitch",
        "gender",
    ],
)


def extract_features(
    gender: Union[str, int],
    transcript: str,
    audio_path: str = None,
    audio_array: np.array = None,
    sampling_frequency: float = None,
):
    try:
        if gender == "male" or gender == 0:
            pitch_floor = 75
            pitch_ceiling = 250
        elif gender == "female" or gender == 1:
            pitch_floor = 100
            pitch_ceiling = 400
        elif gender == "other":
            pitch_floor = 75
            pitch_ceiling = 400
        else:
            raise ValueError(f"Gender value {gender} not recognized.")

        # soundfile = os.path.join(wav_directory, talk_id + ".wav")
        # sound = pm.Sound(soundfile)
        # pcm_data, sample_rate = read_wave(soundfile)

        if audio_path:
            sound = pm.Sound(
                audio_path
            )  # when loading the file parselmouth infers sampling frequency
            pcm_data, sample_rate = librosa.load(
                audio_path, sr=sound.sampling_frequency
            )
        else:
            # audio was already loaded by HF. We can instantiate pm.Sound giving the array
            # See for more details: https://github.com/YannickJadoul/Parselmouth/blob/master/src/parselmouth/Sound.cpp#L194
            sound = pm.Sound(audio_array, sampling_frequency=sampling_frequency)
            pcm_data = audio_array
            sample_rate = sampling_frequency

        # duration = len(pcm_data) / sample_rate # seconds = sample / Hz
        duration = sound.duration
        # print(sample_rate, duration, sound.sampling_frequency)
        sample_rate = int(sample_rate)  # required by VAD

        # print(pcm_data[:10], type(pcm_data[0]))
        if duration > 0.6:
            # print(f"Processing {idx}")

            intensity = pm.praat.call(sound, "To Intensity", pitch_floor, 0.0, True)
            pitch = pm.praat.call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
            vad = webrtcvad.Vad(2)
            frames = list(frame_generator(20, pcm_data, sample_rate))

            # print(len(frames))
            # print(frames)

            silence_intervals, non_silence_intervals = vad_collector(
                sample_rate, duration, 20, vad, frames, 0.1
            )

            speaking_rate = get_speaking_rate(transcript, duration)
            articulatory_rate = get_articulatory_rate(
                transcript, duration, silence_intervals
            )
            snr = get_snr(sound, silence_intervals, non_silence_intervals)

            min_intensity = get_min_intensity(intensity, non_silence_intervals)
            max_intensity = get_max_intensity(intensity, non_silence_intervals)
            mean_intensity = get_mean_intensity(intensity, non_silence_intervals)
            sd_intensity = get_sd_intensity(intensity, non_silence_intervals)

            min_pitch = pm.praat.call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
            max_pitch = pm.praat.call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
            mean_pitch = pm.praat.call(pitch, "Get mean", 0, 0, "Hertz")
            sd_pitch = pm.praat.call(pitch, "Get standard deviation", 0, 0, "Hertz")

        else:
            speaking_rate = None
            articulatory_rate = None
            snr = None

            min_intensity = None
            max_intensity = None
            mean_intensity = None
            sd_intensity = None

            min_pitch = None
            max_pitch = None
            mean_pitch = None
            sd_pitch = None

    except Exception as e:
        # print("New exception")
        # print(e)
        #     return None
        # else:
        speaking_rate = None
        articulatory_rate = None
        snr = None

        min_intensity = None
        max_intensity = None
        mean_intensity = None
        sd_intensity = None

        min_pitch = None
        max_pitch = None
        mean_pitch = None
        sd_pitch = None

    return AudioFeatures(
        speaking_rate,
        articulatory_rate,
        snr,
        min_intensity,
        max_intensity,
        mean_intensity,
        sd_intensity,
        min_pitch,
        max_pitch,
        mean_pitch,
        sd_pitch,
        gender,
    )


SPLITS = ["train", "validation", "test"]


def main(
    dataset,
    lang,
    output_dir,
    num_workers: int = 1,
    reference_col: str = "sentence",
    target_col: str = "gender",
):
    dataset_id = dataset.replace("/", "--")
    is_local_mozilla = False
    if "mozilla" in dataset and lang not in LANGS_TO_LOAD_REMOTE:
        is_local_mozilla = True

    if is_local_mozilla:
        dataset = MozillaCVDataset(
            "/data/milanlp/attanasiog/fair_asr/cv-corpus-16.0-2023-12-06",
            lang,
            "all",
            decode_audio=True,
        )

        dataset.validate_audio(n_jobs=num_workers)
        print("Validity check:", dataset.data["is_valid"].value_counts())
        raw_data = dataset.data.loc[dataset.data["is_valid"] == True]
        # filter rows with nan gender
        # raw_data = dataset.data.loc[dataset.data[target_col].notna()]

    else:
        lang_code = lang if "fleurs" not in dataset else LANG_TO_CONFIG_MAPPING[lang]

        to_cat = list()
        for s in SPLITS:
            d = load_dataset(dataset, lang_code, split=s, trust_remote_code=True)
            d = d.add_column("split", [s] * len(d))
            to_cat.append(d)

        # if split == "all":
        raw_data = concatenate_datasets(to_cat)

        # if "fleurs" in dataset:
        #     print(f"Processing FLEURS: mapping gender IDs with {ID_2_GENDER_MAP}")
        #     raw_data = raw_data.map(lambda x: {target_col: ID_2_GENDER_MAP[x]})

        print(f"Columns of remote dataset {raw_data}")

    print("Config:", dataset_id, lang, output_dir, num_workers, reference_col)
    print("Loading finished.")

    # raw_data = raw_data.head(100)

    #     def extract_features(
    #         gender: str,
    #         transcript: str,
    #         audio_path: str = None,
    #         audio_array: np.array = None,
    #         sampling_frequency: float = None
    # ):

    # TODO: if raw_data is a pandas dataframe the code breaks

    if not is_local_mozilla:
        audio_features = joblib.Parallel(
            n_jobs=num_workers, verbose=5, batch_size=1000
        )(
            joblib.delayed(extract_features)(
                row["gender"],  # shared among all datasets
                row[reference_col],  # transcript column indexed by reference_col
                # We need to differentiate based on local/remote loading
                None,
                row["audio"]["array"],
                row["audio"]["sampling_rate"],
            )
            for row in raw_data
        )
    else:
        audio_features = joblib.Parallel(
            n_jobs=num_workers, verbose=5, batch_size=1000
        )(
            joblib.delayed(extract_features)(
                row["gender"],  # shared among all datasets
                row[reference_col],  # transcript column indexed by reference_col
                # We need to differential based on local/remote loading
                row["audio"],
                None,
                None,
            )
            for idx, row in raw_data.iterrows()
        )

    # print("Initial dataset len:", len(audio_features))
    # audio_features = [a for a in audio_features if a]
    # print("Final len (we couldnt load some of the files:", len(audio_features))

    df = pd.DataFrame(audio_features)
    df["rid"] = list(range(len(audio_features)))
    df["split"] = raw_data["split"].values if is_local_mozilla else raw_data["split"]
    df.set_index("rid").to_csv(f"{output_dir}/af_{dataset_id}_all_{lang}.csv")

    print("Distribution of failed rows:")
    print(df["speaking_rate"].isna().value_counts(normalize=True))


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    print(f"ELAPSED {int(time.time() - stime)} seconds.")
