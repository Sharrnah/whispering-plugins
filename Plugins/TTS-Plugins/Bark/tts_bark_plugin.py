# ============================================================
# Bark Text to Speech Plugin for Whispering Tiger
# V0.3.29
# Bark: https://github.com/suno-ai/bark
# Whispering Tiger: https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import base64
import io
import json
import random
import subprocess
import sys
import time

from importlib import util
import importlib
import pkgutil

import numpy as np
import torch
import torchaudio

import Plugins

from scipy.io.wavfile import write as write_wav

from pathlib import Path
import os

import audio_tools
import settings
import websocket
import downloader
import Models.STT.faster_whisper as faster_whisper


# from Models.STS import DeepFilterNet
# from df.enhance import enhance, init_df, load_audio


def load_module(package_dir, recursive=False):
    package_dir = os.path.abspath(package_dir)
    package_name = os.path.basename(package_dir)

    # Add the parent directory of the package to sys.path
    parent_dir = os.path.dirname(package_dir)
    sys.path.insert(0, parent_dir)

    # Load the package
    spec = util.find_spec(package_name)
    if spec is None:
        raise ImportError(f"Cannot find package '{package_name}'")

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if recursive:
        # Recursively load all submodules
        for _, name, _ in pkgutil.walk_packages(module.__path__, module.__name__ + '.'):
            importlib.import_module(name)

    # Remove the parent directory from sys.path
    sys.path.pop(0)

    return module


def estimate_remaining_time(total_segments, segment_times, segments_for_estimate=3, last_x_segments=None):
    """
    Estimates the remaining time based on the average time of specified segment times.

    Parameters:
    total_segments (int): Total number of segments.
    segment_times (list): List of times taken for each segment.
    segments_for_estimate (int): Minimum number of segments needed to start estimating.
    last_x_segments (int, optional): Number of recent segments to use for estimating the average time.
                                     If None, all available segments are used.

    Returns:
    str: Formatted string of estimated remaining time or "[estimating...]" if not enough data.
    """
    if len(segment_times) < segments_for_estimate:
        return " [estimating...]"

    if last_x_segments is not None:
        # Use only the last x segments for estimation
        relevant_segment_times = segment_times[-last_x_segments:]
    else:
        # Use all available segment times for estimation
        relevant_segment_times = segment_times

    # Calculate the average time per segment
    avg_time_per_segment = sum(relevant_segment_times) / len(relevant_segment_times)

    # Calculate the remaining segments
    remaining_segments = total_segments - len(segment_times)

    # Estimate the remaining time
    estimated_remaining_time = avg_time_per_segment * remaining_segments

    # Convert estimated time to hours, minutes, and seconds
    hours, rem = divmod(estimated_remaining_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Format the estimated time
    estimated_time_str = ""
    if hours > 0:
        estimated_time_str += f"{int(hours)} hrs. "
    if minutes > 0:
        estimated_time_str += f"{int(minutes)} min. "
    estimated_time_str += f"{int(seconds)} sec."

    if estimated_time_str:
        return f" [~ {estimated_time_str} remaining]"
    else:
        return ""


def calculate_total_time(segment_times):
    """
    Calculates the total time taken for all segments.

    Parameters:
    segment_times (list): List of times taken for each segment.

    Returns:
    str: Formatted string of total time taken.
    """
    total_time = sum(segment_times)

    # Convert total time to hours, minutes, and seconds
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Format the total time
    total_time_str = ""
    if hours > 0:
        total_time_str += f"{int(hours)} hrs. "
    if minutes > 0:
        total_time_str += f"{int(minutes)} min. "
    total_time_str += f"{int(seconds)} sec."
    if total_time_str:
        return f" [{total_time_str} total]"
    else:
        return ""


bark_plugin_dir = Path(Path.cwd() / "Plugins" / "bark_plugin")
os.makedirs(bark_plugin_dir, exist_ok=True)

einops_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/68/24/b05452c986e8eff11f47e123a40798ae693f2fa1ed2f9546094997d2f6be/einops-0.6.1-py3-none-any.whl",
    "sha256": "99149e46cc808956b174932fe563d920db4d6e5dadb8c6ecdaa7483b7ef7cfc3",
    "path": "einops"
}

encodec_dependency = {
    "url": "https://files.pythonhosted.org/packages/62/59/e47bbd0542d0e6f4ce9983d5eb458a01d4b42c81e5c410cb9e159b1061ae/encodec-0.1.1.tar.gz",
    "sha256": "36dde98ccfe6c51a15576476cadfcb3b35a63507b8b8555abd69889a6fba6772",
    "path": "encodec-0.1.1/encodec"
}

funcy_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/d5/08/c2409cb01d5368dcfedcbaffa7d044cc8957d57a9d0855244a5eb4709d30/funcy-2.0-py2.py3-none-any.whl",
    "sha256": "53df23c8bb1651b12f095df764bfb057935d49537a56de211b098f4c79614bb0",
    "path": "funcy"
}

bark_dependency_module = {
    "url": "https://github.com/Sharrnah/bark-with-voice-clone/archive/99f40108dcc6c68cbaa6a6a58a0510f714d6266c.zip",
    "sha256": "f5e2f04306e576b00815927be4302e6833f25b6d9f38d42a295f20bdea663812",
    "path": "bark-with-voice-clone-99f40108dcc6c68cbaa6a6a58a0510f714d6266c",
}

vocos_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/98/b3/445694d1059688a76a997c61936fef938b7d90f905a00754b4a441e7fcbd/vocos-0.0.3-py3-none-any.whl",
    "sha256": "0578b20b4ba57533a9d9b3e5ec3f81982f6fabd07ef02eb175fa9ee5da1e3cac",
    "path": "vocos"
}

# used for optional audio normalization
pyloudnorm_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/58/f5/6724805521ab4e723a12182f92374031032aff28a8a89dc8505c52b79032/pyloudnorm-0.1.1-py3-none-any.whl",
    "sha256": "d7f12ebdd097a464d87ce2878fc4d942f15f8233e26cc03f33fefa226f869a14",
    "path": "pyloudnorm"
}
# pyloudnorm dependency future
pyloudnorm_future_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/9e/cf/95b17d4430942dbf291fa5411d8189374a2e6dba91d9ef077e7fb8e869bc/future-0.18.0-cp36-none-any.whl",
    "sha256": "3f9c52f6c3f4e287bdd9b13de6cfd72373fb694aa391b5e511deef3db15d6a62",
    "path": "future"
}

bark_voice_clone_tool = {
    "urls": [
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:projects/voice-cloning-bark/barkVoiceClone_v0.0.2.zip",
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:projects/voice-cloning-bark/barkVoiceClone_v0.0.2.zip"
    ],
    "sha256": "a32afb7a2a9e4b706ecfeefb9a010e6e975096e7573ec8ab70fefbdfaabd4bd3",
    "path": "barkVoiceClone",
}


class BarkTTSPlugin(Plugins.Base):
    bark_module = None
    sample_rate = 24000

    pyloudnorm_module = None

    encodec = None
    hubert_module = None
    hubert_model = None
    hubert_tokenizer = None
    vocos = None

    audio_enhancer = None

    transcript_model = None

    stop_batch_processing = False

    def get_plugin(self, class_name):
        for plugin_inst in Plugins.plugins:
            if plugin_inst.__class__.__name__ == class_name:
                return plugin_inst  # return plugin instance
        return None

    # Function to calculate LUFS
    def calculate_lufs(self, audio, sample_rate):
        meter = self.pyloudnorm_module.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(audio)
        return loudness

    # Function to normalize the audio based on LUFS
    def normalize_audio_lufs(self, audio, sample_rate, lower_threshold=-24.0, upper_threshold=-16.0, gain_factor=2.0):
        lufs = self.calculate_lufs(audio, sample_rate)

        print(f"LUFS: {lufs}")

        # If LUFS is lower than the lower threshold, increase volume
        if lufs < lower_threshold:
            print(f"audio is too quiet, increasing volume")
            gain = (lower_threshold - lufs) / gain_factor
            audio = audio * np.power(10.0, gain / 20.0)

        # If LUFS is higher than the upper threshold, decrease volume
        elif lufs > upper_threshold:
            print(f"audio is too loud, decreasing volume")
            gain = (upper_threshold - lufs) * gain_factor
            audio = audio * np.power(10.0, gain / 20.0)

        # Limit audio values to [-1, 1] (this is important to avoid clipping when converting to 16-bit PCM)
        audio = np.clip(audio, -1, 1)

        return audio, lufs

    def trim_silence(self, audio, silence_threshold=0.01):
        # Compute absolute value of audio waveform
        audio_abs = np.abs(audio)

        # Find the first index where the absolute value of the waveform exceeds the threshold
        start_index = np.argmax(audio_abs > silence_threshold)

        # Reverse the audio waveform and do the same thing to find the end index
        end_index = len(audio) - np.argmax(audio_abs[::-1] > silence_threshold)

        # If start_index is not 0, some audio at the start has been trimmed
        if start_index > 0:
            print(f"Trimmed {start_index} samples from the start of the audio")

        # If end_index is not the length of the audio, some audio at the end has been trimmed
        if end_index < len(audio):
            print(f"Trimmed {len(audio) - end_index} samples from the end of the audio")

        # Return the trimmed audio
        return audio[start_index:end_index]

    def remove_silence_parts(self, audio, sample_rate, silence_threshold=0.01, max_silence_length=1.1,
                             keep_silence_length=0.06):
        audio_abs = np.abs(audio)
        above_threshold = audio_abs > silence_threshold

        # Convert length parameters to number of samples
        max_silence_samples = int(max_silence_length * sample_rate)
        keep_silence_samples = int(keep_silence_length * sample_rate)

        last_silence_end = 0
        silence_start = None

        chunks = []

        for i, sample in enumerate(above_threshold):
            if not sample:
                if silence_start is None:
                    silence_start = i
            else:
                if silence_start is not None:
                    silence_duration = i - silence_start
                    if silence_duration > max_silence_samples:
                        # Subtract keep_silence_samples from the start and add it to the end
                        start = max(last_silence_end - keep_silence_samples, 0)
                        end = min(silence_start + keep_silence_samples, len(audio))
                        chunks.append(audio[start:end])
                        last_silence_end = i
                    silence_start = None

        # Append the final chunk of audio after the last silence
        if last_silence_end < len(audio):
            start = max(last_silence_end - keep_silence_samples, 0)
            end = len(audio)
            chunks.append(audio[start:end])

        if len(chunks) == 0:
            print("No non-silent sections found in audio.")
            return np.array([])
        else:
            print(f"found {len(chunks)} non-silent sections in audio")
            return np.concatenate(chunks)

    def _load_vocos_model(self):
        # load the vocos module (optional vocoder)
        if self.get_plugin_setting("use_vocos", True) and self.vocos is None:
            if not Path(bark_plugin_dir / vocos_dependency_module["path"] / "__init__.py").is_file():
                downloader.download_extract([vocos_dependency_module["url"]],
                                            str(bark_plugin_dir.resolve()),
                                            vocos_dependency_module["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(bark_plugin_dir / os.path.basename(vocos_dependency_module["url"])),
                                                str(bark_plugin_dir.resolve()),
                                            ),
                                            title="Bark - vocos module", extract_format="zip")
            vocos_module = load_module(str(Path(bark_plugin_dir / vocos_dependency_module["path"]).resolve()))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.vocos = vocos_module.Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device)

    def _levenshtein_distance(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def _search_word_levenshtein(self, original_text, generated_text, threshold=2):
        # Remove punctuation
        for char in [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "\"", "'", "-", "_", "=", "+", "*", "/", "\\", "|", "<", ">", "\n", "\t", "\r"]:
            generated_text = generated_text.replace(char, " ")
            original_text = original_text.replace(char, " ")

        # Convert to lowercase and split the text into words while removing extra spaces
        generated_text_words = [word for word in generated_text.lower().split()]
        original_text_words = [word for word in original_text.lower().split()]

        total_min_distance = 0
        num_words = len(original_text_words)

        # Search for the best match for each word in the original text
        for single_orig_word in original_text_words:
            word_min_distance = float('inf')  # Start with a large number

            # Compare with each word in the generated text
            for single_gen_word in generated_text_words:
                current_distance = self._levenshtein_distance(single_gen_word, single_orig_word)
                if current_distance < word_min_distance:
                    word_min_distance = current_distance

            # Aggregate the minimum distance for each word
            total_min_distance += word_min_distance

        # Calculate average minimum distance per word
        average_min_distance = total_min_distance / num_words if num_words > 0 else float('inf')

        # Compare average minimum distance to the threshold
        return average_min_distance <= threshold

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                # "history_prompt": None,
                "history_prompt": {"type": "file_open", "accept": ".npz", "value": ""},
                "prompt_wrap": "##",
                "seed": -1,

                "long_text": False,
                "long_text_stable_frequency": {"type": "slider", "min": 0, "max": 10, "step": 1, "value": 1},
                "long_text_split_pause": {"type": "slider", "min": 0, "max": 5, "step": 0.01, "value": 0},
                "split_character_goal_length": {"type": "slider", "min": 1, "max": 300, "step": 1, "value": 130},
                "split_character_max_length": {"type": "slider", "min": 1, "max": 300, "step": 1, "value": 170},
                "split_character_jitter": {"type": "slider", "min": 0, "max": 100, "step": 1, "value": 0},
                "use_previous_history_for_last_segment": False,
                "long_text_stable_frequency_info": {
                    "label": "stable_frequency\n0 = each continuation uses the history prompt of the previous.\n1 = each generation uses same history prompt.\n2+ = each *n generation uses the first history prompt.",
                    "type": "label", "style": "left"},
                "custom_split_characters": ",",

                "use_offload_cpu": True,
                "use_small_models": True,
                "use_half_precision": False,
                #"use_mps": False,
                "use_gpu": False,
                "use_vocos": True,
                "use_vocos_on_result": True,

                "temperature_text": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.7},
                "temperature_waveform": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.7},
                "min_eos_p": {"type": "slider", "min": 0.01, "max": 1.0, "step": 0.01, "value": 0.05},
                "min_eos_p_info": {
                    "label": "min_eos_p - min. end of sentence probability (default: 0.05).\nLower = speech generation ends earlier.\nCan reduce additional words at the end.",
                    "type": "label", "style": "left"},

                "write_last_history_prompt": False,
                "write_last_history_prompt_file": {"type": "file_save", "accept": ".npz",
                                                   "value": "bark_prompts/last_prompt.npz"},

                "batch_size": 1,
                "batch_prompts": {"type": "textarea", "rows": 5, "value": ""},
                "batch_folder": {"type": "folder_open", "accept": "", "value": "bark_prompts/multi_generations"},
                "zz_batch_button": {"label": "Batch Generate", "type": "button", "style": "primary"},
                "zz_batch_stop_button": {"label": "Stop Batch Generate", "type": "button", "style": "default"},

                "clone_voice_audio_filepath": {"type": "file_open", "accept": ".wav",
                                               "value": "bark_clone_voice/clone_voice.wav"},
                "clone_voice_prompt": "",
                "zz_clone_voice_button": {"label": "Start Voice Clone", "type": "button", "style": "primary"},

                "zz_clone_voice_better_button": {"label": "Start Voice Clone", "type": "button", "style": "primary"},
                "zzz_clone_voice_better_info": {
                    "label": "To get a more stable voice, use the cloned *.npz file as history prompt\nand use Batch Processing with that to find a similar voice.",
                    "type": "label", "style": "center"},

                "normalize": True,
                "normalize_lower_threshold": -24.0,
                "normalize_upper_threshold": -16.0,
                "normalize_gain_factor": 1.3,
                # "audio_denoise": True,

                "trim_silence": True,
                "remove_silence_parts": True,
                "silence_threshold": {"type": "slider", "min": 0.0, "max": 2.0, "step": 0.01, "value": 0.03},
                "max_silence_length": {"type": "slider", "min": 0.0, "max": 3.0, "step": 0.1, "value": 0.8},
                "keep_silence_length": {"type": "slider", "min": 0.0, "max": 3.0, "step": 0.01, "value": 0.20},

                "vocos_file": {"type": "file_open", "accept": ".wav",
                               "value": "bark_clone_voice/clone_voice.wav"},
                "vocos_file_button": {"label": "apply Vocos to file", "type": "button", "style": "default"},

                "validate_generated_audio": False,
                "validate_max_distance_threshold": {"type": "slider", "min": 0, "max": 20, "step": 1, "value": 2},
                "validate_max_retries": {"type": "slider", "min": 0, "max": 20, "step": 1, "value": 3},
            },
            settings_groups={
                "General": ["history_prompt", "prompt_wrap", "temperature_text", "temperature_waveform", "min_eos_p",
                            "min_eos_p_info", "seed",
                            "validate_generated_audio", "validate_max_distance_threshold", "validate_max_retries"],
                "Long Text Gen.": ["long_text", "long_text_stable_frequency", "long_text_stable_frequency_info",
                                   "long_text_split_pause", "split_character_goal_length", "split_character_max_length",
                                   "split_character_jitter", "use_previous_history_for_last_segment", "custom_split_characters"],
                "History Prompt": ["write_last_history_prompt", "write_last_history_prompt_file"],
                "Voice Cloning": ["clone_voice_audio_filepath", "clone_voice_prompt", "zz_clone_voice_button"],
                "Model Settings": ["use_offload_cpu", "use_small_models", "use_gpu", "use_vocos",
                                   "use_vocos_on_result", "use_half_precision"],
                                    # "use_mps"
                "Voice Cloning Better": ["clone_voice_audio_filepath", "zz_clone_voice_better_button",
                                         "zzz_clone_voice_better_info"],
                "Batch Processing": ["batch_prompts", "batch_size", "batch_folder", "zz_batch_button", "zz_batch_stop_button"],
                "Audio Processing": ["normalize", "normalize_lower_threshold", "normalize_upper_threshold",
                                     "normalize_gain_factor", "vocos_file_button", "vocos_file"],
                "Audio Processing 2": ["trim_silence", "remove_silence_parts", "silence_threshold",
                                       "max_silence_length", "keep_silence_length"],
            }
        )

        if self.is_enabled(False):
            model_cache_dir = Path(bark_plugin_dir / "bark_models")

            # disable default tts engine
            settings.SetOption("tts_enabled", False)

            # torch backend settings
            torch.backends.cuda.matmul.allow_tf32 = True

            seed = self.get_plugin_setting("seed")
            if seed is not None and seed >= 0:
                # make pytorch fully deterministic (disabling CuDNN benchmarking can slow down computations)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            else:
                # disable deterministic
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

            use_offload_cpu = self.get_plugin_setting("use_offload_cpu", True)

            os.environ["SUNO_OFFLOAD_CPU"] = str(use_offload_cpu)
            if self.get_plugin_setting("use_small_models", True):
                os.environ["USE_SMALL_MODELS"] = str(self.get_plugin_setting("use_small_models", True))

            os.environ["CACHE_DIR"] = str(model_cache_dir.resolve())

            #os.environ["SUNO_ENABLE_MPS"] = str(self.get_plugin_setting("use_mps", False))
            use_gpu = self.get_plugin_setting("use_gpu", True)

            # set cache directory
            #os.environ["XDG_CACHE_HOME"] = str(Path(bark_plugin_dir / ".cache").resolve())

            websocket.set_loading_state("bark_plugin_loading", True)

            # load the einops module
            einops_path = Path(bark_plugin_dir / einops_dependency_module["path"])
            if not Path(einops_path / "__init__.py").is_file():
                downloader.download_extract([einops_dependency_module["url"]],
                                            str(bark_plugin_dir.resolve()),
                                            einops_dependency_module["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(bark_plugin_dir / os.path.basename(
                                                    einops_dependency_module["url"])),
                                                str(bark_plugin_dir.resolve()),
                                            ),
                                            title="Bark - einops module", extract_format="zip")
            einops = load_module(str(Path(bark_plugin_dir / einops_dependency_module["path"]).resolve()))

            # load the encodec module
            encodec_path = Path(bark_plugin_dir / encodec_dependency["path"])
            if not Path(encodec_path / "__init__.py").is_file():
                downloader.download_extract([encodec_dependency["url"]],
                                            str(bark_plugin_dir.resolve()),
                                            encodec_dependency["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_tar_gz,
                                            fallback_extract_func_args=(
                                                str(bark_plugin_dir / os.path.basename(encodec_dependency["url"])),
                                                str(bark_plugin_dir.resolve()),
                                            ),
                                            title="Bark - encodec module", extract_format="tar.gz")
            self.encodec = load_module(str(Path(bark_plugin_dir / encodec_dependency["path"]).resolve()))

            # load the funcy module
            if not Path(bark_plugin_dir / funcy_dependency_module["path"] / "__init__.py").is_file():
                downloader.download_extract([funcy_dependency_module["url"]],
                                            str(bark_plugin_dir.resolve()),
                                            funcy_dependency_module["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(bark_plugin_dir / os.path.basename(funcy_dependency_module["url"])),
                                                str(bark_plugin_dir.resolve()),
                                            ),
                                            title="Bark - funcy module", extract_format="zip")
            funcy = load_module(str(Path(bark_plugin_dir / funcy_dependency_module["path"]).resolve()))

            # load the vocos module (optional vocoder)
            self._load_vocos_model()

            # load the bark module
            if not Path(bark_plugin_dir / bark_dependency_module["path"] / "bark" / "__init__.py").is_file():
                downloader.download_extract([bark_dependency_module["url"]],
                                            str(bark_plugin_dir.resolve()),
                                            bark_dependency_module["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(bark_plugin_dir / os.path.basename(bark_dependency_module["url"])),
                                                str(bark_plugin_dir.resolve()),
                                            ),
                                            title="Bark - bark module", extract_format="zip")

            self.bark_module = load_module(
                str(Path(bark_plugin_dir / bark_dependency_module["path"] / "bark").resolve()))

            # load the future module
            future_path = Path(bark_plugin_dir / pyloudnorm_future_dependency_module["path"])
            if not Path(future_path / "__init__.py").is_file():
                downloader.download_extract([pyloudnorm_future_dependency_module["url"]],
                                            str(bark_plugin_dir.resolve()),
                                            pyloudnorm_future_dependency_module["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(bark_plugin_dir / os.path.basename(
                                                    pyloudnorm_future_dependency_module["url"])),
                                                str(bark_plugin_dir.resolve()),
                                            ),
                                            title="Bark - pyloudnorm future module", extract_format="zip")
            future = load_module(str(Path(bark_plugin_dir / pyloudnorm_future_dependency_module["path"]).resolve()))

            # load the audio normalization module
            if not Path(bark_plugin_dir / pyloudnorm_dependency_module["path"] / "__init__.py").is_file():
                downloader.download_extract([pyloudnorm_dependency_module["url"]],
                                            str(bark_plugin_dir.resolve()),
                                            pyloudnorm_dependency_module["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(bark_plugin_dir / os.path.basename(
                                                    pyloudnorm_dependency_module["url"])),
                                                str(bark_plugin_dir.resolve()),
                                            ),
                                            title="Bark - pyloudnorm module", extract_format="zip")
            self.pyloudnorm_module = load_module(
                str(Path(bark_plugin_dir / pyloudnorm_dependency_module["path"]).resolve()))

            # download and load all models
            use_small_models = self.get_plugin_setting("use_small_models", True)
            print("download and load all bark models", ("small" if use_small_models else "large"))
            if use_offload_cpu:
                self.bark_module.load_model(
                    model_type="text", use_gpu=use_gpu, use_small=use_small_models, force_reload=False, path=str(model_cache_dir.resolve())
                )
            else:
                self.bark_module.preload_models(
                    text_use_gpu=use_gpu,
                    text_use_small=use_small_models,
                    coarse_use_gpu=use_gpu,
                    coarse_use_small=use_small_models,
                    fine_use_gpu=use_gpu,
                    fine_use_small=use_small_models,
                    codec_use_gpu=use_gpu,
                    path=str(model_cache_dir.resolve()),
                )

            if use_gpu and self.get_plugin_setting("use_half_precision"):
                self.bark_module.models_to(torch.float16)

            print("bark models loaded")
            websocket.set_loading_state("bark_plugin_loading", False)

        pass

    def transcribe(self, audio_data):
        compute_dtype = settings.GetOption("whisper_precision")
        model = settings.GetOption("model")
        ai_device = settings.GetOption("ai_device")
        whisper_language = self.get_plugin_setting("language_spoken")
        if whisper_language == "auto":
            whisper_language = None

        if self.transcript_model is None:
            self.transcript_model = faster_whisper.FasterWhisper(model, device=ai_device, compute_type=compute_dtype,
                                                                 cpu_threads=2, num_workers=2)

        if self.transcript_model is not None:
            try:
                audio_data = np.int16(audio_data * 32767)  # Convert to 16-bit PCM
                buff = io.BytesIO()
                write_wav(buff, self.sample_rate, audio_data)
                buff.seek(0)

                result = self.transcript_model.transcribe(buff, task="transcribe",
                                                                      language=whisper_language,
                                                                      condition_on_previous_text=True,
                                                                      initial_prompt=None,
                                                                      logprob_threshold=-1.0,
                                                                      no_speech_threshold=0.6,
                                                                      temperature=[
                                                                          0.0,
                                                                          0.2,
                                                                          0.4,
                                                                          0.6,
                                                                          0.8,
                                                                          1.0,
                                                                      ],
                                                                      beam_size=5,
                                                                      word_timestamps=False,
                                                                      without_timestamps=False,
                                                                      patience=1.0
                                                                      )

                return result.get('text').strip()

            except Exception as e:
                print(e)

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def audio_processing(self, audio_data, skip_infinity_lufs=True):
        # Normalize audio
        if self.get_plugin_setting("normalize", True):
            lower_threshold = self.get_plugin_setting("normalize_lower_threshold", -24.0)
            upper_threshold = self.get_plugin_setting("normalize_upper_threshold", -16.0)
            gain_factor = self.get_plugin_setting("normalize_gain_factor", 1.3)
            audio_data, lufs = self.normalize_audio_lufs(audio_data, self.sample_rate, lower_threshold, upper_threshold,
                                                         gain_factor)
            if lufs == float('-inf') and skip_infinity_lufs:
                print("Audio seems to be unusable. skipping")
                return None

        # Trim silence
        if self.get_plugin_setting("trim_silence", True):
            audio_data = self.trim_silence(audio_data)

        # Remove silence parts
        if self.get_plugin_setting("remove_silence_parts", False):
            silence_threshold = self.get_plugin_setting("silence_threshold")
            keep_silence_length = self.get_plugin_setting("keep_silence_length")
            max_silence_length = self.get_plugin_setting("max_silence_length")
            audio_data = self.remove_silence_parts(audio_data, self.sample_rate, silence_threshold=silence_threshold,
                                                   keep_silence_length=keep_silence_length,
                                                   max_silence_length=max_silence_length)

        # return early if no audio data
        if len(audio_data) == 0:
            return None

        return audio_data


    def _split_segment(self, segment, goal_length, custom_chars, valid_ending_chars):
        # Function to split segments that are too long
        segments = []
        while len(segment) > goal_length:
            split_point = -1
            # Look for custom split characters first
            if custom_chars:
                split_points = [segment.rfind(char, 0, goal_length) for char in custom_chars]
                split_points = [p for p in split_points if p != -1]
                if split_points:
                    split_point = max(split_points) + 1  # Include the split character

            # No custom split point found, find nearest space
            if split_point == -1:
                split_point = segment.rfind(' ', 0, goal_length)
                if split_point == -1:
                    split_point = goal_length  # Default to goal length if no space

            new_segment = segment[:split_point].strip()
            segment = segment[split_point:].strip()
            if new_segment:
                segments.append(new_segment)

        if segment:  # Add the last part if not empty
            segments.append(segment)
        return segments

    def chunk_up_text(self, text_prompt, split_character_goal_length, split_character_max_length,
                      split_character_jitter=0, custom_split_chars=""):

        if split_character_jitter > 0:
            split_character_goal_length = random.randint(split_character_goal_length - split_character_jitter,
                                                         split_character_goal_length + split_character_jitter)
            split_character_max_length = random.randint(split_character_max_length - split_character_jitter,
                                                        split_character_max_length + split_character_jitter)

        audio_segments = self.bark_module.split_general_purpose(text_prompt,
                                                                split_character_goal_length=split_character_goal_length,
                                                                split_character_max_length=split_character_max_length)

        print(f"Splitting long text aiming for {split_character_goal_length} chars max {split_character_max_length}")

        valid_ending_chars = ".;!?\n\"" + custom_split_chars

        i = 0
        while i < len(audio_segments):
            if not audio_segments[i][-1] in valid_ending_chars:
                if any(char in custom_split_chars for char in audio_segments[i]) and custom_split_chars:
                    # Combine the current segment with the next, if applicable.
                    if i < len(audio_segments) - 1:  # Make sure there is a next segment to combine with.
                        audio_segments[i] += ' ' + audio_segments[i + 1]
                        audio_segments.pop(i + 1)
                    continue  # Skip the increment at the end of the loop to recheck this now extended segment.
            i += 1  # Only increment here if the segment ends properly or there are no custom chars to merge.

        # Split long segments further if needed
        final_segments = []
        i = 0
        while i < len(audio_segments):
            current_segment = audio_segments[i]

            # Check if we can merge the current segment with the next
            if i < len(audio_segments) - 1 and current_segment[-1] not in valid_ending_chars:
                next_segment = audio_segments[i + 1]
                combined_segment = current_segment + " " + next_segment
                if len(combined_segment) <= split_character_max_length:
                    audio_segments[i] = combined_segment  # Merge segments
                    audio_segments.pop(i + 1)  # Remove the merged segment
                    continue  # Reevaluate the current position
                else:
                    # Attempt to split the combined segment if alone it's too long
                    if len(current_segment) > split_character_max_length:
                        current_segment = self._split_segment(current_segment, split_character_goal_length, custom_split_chars, valid_ending_chars)

            # If the segment is too long, split it
            elif len(current_segment) > split_character_max_length:
                current_segment = self._split_segment(current_segment, split_character_goal_length, custom_split_chars, valid_ending_chars)

            # Add the current or modified segment to final segments if it's valid
            if current_segment:
                final_segments.extend(current_segment if isinstance(current_segment, list) else [current_segment])
            i += 1

        return final_segments


    def generate_segments_vocos(self, text, text_temp=0.7, waveform_temp=0.7, min_eos_p=0.05,
                                history_prompt=None, silent=False):
        semantic_tokens = self.bark_module.text_to_semantic(text, history_prompt=history_prompt,
                                                            temp=text_temp, min_eos_p=min_eos_p, silent=silent, )
        history_prompt_data = self.bark_module.semantic_to_audio_tokens(
            semantic_tokens, history_prompt=history_prompt, temp=waveform_temp, silent=silent,
            output_full=True,
        )

        # reconstruct with Vocos
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        audio_tokens_torch = torch.from_numpy(history_prompt_data["fine_prompt"]).to(device)
        features = self.vocos.codes_to_features(audio_tokens_torch)
        audio_data_np_array = self.vocos.decode(features,
                                                bandwidth_id=torch.tensor([2], device=device)).cpu().numpy()  # 6 kbps

        audio_data_np_array = audio_tools.resample_audio(audio_data_np_array, 24000, 24000, target_channels=1,
                                                         input_channels=1, dtype="float32")

        ## Ensure audio is mono by checking the shape and adjusting if necessary
        #if len(audio_data_np_array.shape) > 1 and audio_data_np_array.shape[0] != 1:
        #    # Assuming the data is stereo and needs to be converted to mono by averaging the channels
        #    audio_data_np_array = np.mean(audio_data_np_array, axis=0, keepdims=True)

        return history_prompt_data, audio_data_np_array

    def apply_vocos_on_audio(self, audio_data):
        # check if audio_data is bytes
        wav_file = audio_data
        if isinstance(audio_data, bytes):
            wav_file = io.BytesIO(audio_data)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        y, sr = torchaudio.load(wav_file)
        if y.size(0) > 1:  # mix to mono
            y = y.mean(dim=0, keepdim=True)
        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
        y = y.to(device)
        bandwidth_id = torch.tensor([2]).to(device)  # 6 kbps
        y_hat = self.vocos(y, bandwidth_id=bandwidth_id)

        audio_data_np_array = audio_tools.resample_audio(y_hat, 24000, self.sample_rate, target_channels=1,
                                                         input_channels=1, dtype="float32")

        audio_data_16bit = np.int16(audio_data_np_array * 32767)  # Convert to 16-bit PCM

        buff = io.BytesIO()
        write_wav(buff, self.sample_rate, audio_data_16bit)

        buff.seek(0)
        return buff

    def generate_tts(self, text, text_temp=0.7, waveform_temp=0.7, write_last_history_prompt=False,
                     last_hisory_prompt_file=None, prompt_wrap="##", skip_infinity_lufs=True, long_text=False,
                     long_text_stable_frequency=0, long_text_split_pause=0.0, silent=False):
        is_stopping = False
        validate_max_retries = self.get_plugin_setting("validate_max_retries")
        history_prompt = self.get_plugin_setting("history_prompt", None)
        if history_prompt == "":
            history_prompt = None

        worker_seed = self.get_plugin_setting("seed", -1)
        if worker_seed is None or worker_seed <= -1:
            worker_seed = random.randint(0, 2 ** 32 - 1)
            # worker_seed = np.random.default_rng().integers(1, 2**32 - 1)
            if not silent:
                print("Bark: using seed %d" % worker_seed)
        self.set_seed(worker_seed)

        min_eos_p = self.get_plugin_setting("min_eos_p")

        use_vocos = self.get_plugin_setting("use_vocos", True) and self.vocos is not None
        self._load_vocos_model()

        audio_data_np_array = None

        segment_times = []
        if long_text:
            audio_arr_segments = []

            estimated_time = self.bark_module.estimate_spoken_time(text)
            if not silent:
                print(f"estimated_time: {estimated_time}")

            split_character_goal_length = self.get_plugin_setting("split_character_goal_length")
            split_character_max_length = self.get_plugin_setting("split_character_max_length")
            split_character_jitter = self.get_plugin_setting("split_character_jitter")
            use_previous_history_for_last_segment = self.get_plugin_setting("use_previous_history_for_last_segment")
            custom_split_characters = self.get_plugin_setting("custom_split_characters")

            audio_segments = self.chunk_up_text(text,
                                                split_character_goal_length=split_character_goal_length,
                                                split_character_max_length=split_character_max_length,
                                                split_character_jitter=split_character_jitter,
                                                custom_split_chars=custom_split_characters,
                                                )
            total_segments = len(audio_segments)
            if not silent:
                print(f"audio_segments: {total_segments}")

            history_prompt_for_next_segment = history_prompt

            for i, segment_text in enumerate(audio_segments):
                if self.stop_batch_processing:
                    self.stop_batch_processing = False
                    is_stopping = True
                    print("long text generating stopped.")
                    break

                estimated_time = self.bark_module.estimate_spoken_time(segment_text)

                if not silent:
                    print(f"segment: {i+1} of {total_segments}")
                    print(f"estimated_time: {estimated_time}")

                segment_text = prompt_wrap.replace("##", segment_text)

                # disable generate_audio progress bar and display full batch progress instead
                display_long_text_progress = (total_segments >= 3 and i+1 < total_segments)

                if display_long_text_progress and not silent:
                    # Estimate time for the remaining segments after the first three are done
                    estimate_time_full_str = estimate_remaining_time(total_segments, segment_times, 3)
                    print(f"long text progress: {int(i / total_segments * 100)}% ({i} of {total_segments} segments){estimate_time_full_str}")

                # audio_data_np_array = None
                # history_prompt_data = None

                start_time = time.time()
                for retry_num in range(validate_max_retries):

                    if not use_vocos:
                        history_prompt_data, audio_data_np_array = self.bark_module.generate_audio(segment_text,
                                                                                                   history_prompt=history_prompt_for_next_segment,
                                                                                                   text_temp=text_temp,
                                                                                                   waveform_temp=waveform_temp,
                                                                                                   min_eos_p=min_eos_p,
                                                                                                   output_full=True,
                                                                                                   silent=(display_long_text_progress or silent),
                                                                                                   )
                    else:
                        history_prompt_data, audio_data_np_array = self.generate_segments_vocos(segment_text,
                                                                                                text_temp=text_temp,
                                                                                                waveform_temp=waveform_temp,
                                                                                                min_eos_p=min_eos_p,
                                                                                                history_prompt=history_prompt_for_next_segment,
                                                                                                silent=(display_long_text_progress or silent),
                                                                                                )

                    audio_data_np_array = self.audio_processing(audio_data_np_array,
                                                                skip_infinity_lufs=skip_infinity_lufs)

                    if not silent:
                        print(f"attempt: {retry_num} of {validate_max_retries}")

                    if self.get_plugin_setting(
                            "validate_generated_audio") and audio_data_np_array is not None and audio_data_np_array.size > 0:
                        if not silent:
                            print(f"Validating generated audio for segment...")
                        levenshtein_threshold = int(self.get_plugin_setting("validate_max_distance_threshold"))

                        generated_text = self.transcribe(audio_data_np_array)
                        if generated_text is not None and generated_text != "" and self._search_word_levenshtein(segment_text, generated_text,
                                                                                        levenshtein_threshold):
                            if not silent:
                                print("Generated audio is valid")
                            break

                    elif audio_data_np_array is not None and audio_data_np_array.size > 0:  # Check if the array is not empty
                        break

                else:
                    print(f"Failed to generate non-empty audio for segment {i} after {validate_max_retries} retries")
                    continue  # Skip adding this segment to the final array

                if audio_data_np_array is not None:
                    audio_data_np_array = np.int16(audio_data_np_array * 32767)  # Convert to 16-bit PCM
                    audio_arr_segments.append(audio_data_np_array)

                # set history_prompt when it's initially None
                if history_prompt is None and history_prompt_data is not None:
                    history_prompt = history_prompt_data

                # Check if it's the last segment and the setting is enabled
                if use_previous_history_for_last_segment and i == len(audio_segments) - 1:
                    history_prompt_for_next_segment = history_prompt
                # use history prompt in configured frequency
                elif long_text_stable_frequency > 0 and (i + 1) % long_text_stable_frequency == 0:
                    history_prompt_for_next_segment = history_prompt
                else:
                    history_prompt_for_next_segment = history_prompt_data

                end_time = time.time()
                segment_times.append(end_time - start_time)

            if self.get_plugin_setting("use_vocos_on_result"):
                vocos_segments = []
                if not silent:
                    print("applying vocos on result")
                segment_num = 0
                for segment in audio_arr_segments:
                    segment_num += 1
                    vocos_segment = audio_tools.numpy_array_to_wav_bytes(segment, self.sample_rate)
                    vocos_segment = self.apply_vocos_on_audio(vocos_segment)
                    if vocos_segment is not None:
                        vocos_segment = audio_tools.wav_bytes_to_numpy_array(vocos_segment.getvalue())
                        vocos_segments.append(vocos_segment)
                    else:
                        print("could not apply vocos on segment")
                        vocos_segments.append(segment)
                audio_arr_segments = vocos_segments

            # put all audio together
            audio_arr_segments = [seg for seg in audio_arr_segments if seg.size > 0]  # Filter out any empty segments
            if len(audio_arr_segments) > 0 and long_text_split_pause > 0.0:
                audio_with_pauses = []
                #pause_samples = np.zeros(int(long_text_split_pause * self.sample_rate))
                pause_samples = np.zeros(int(long_text_split_pause * self.sample_rate), dtype=np.int16) # add 16bit PCM pause audio
                # Iterate over each audio segment
                for segment in audio_arr_segments:
                    # Add the audio segment
                    audio_with_pauses.append(segment)
                    # Add a pause
                    audio_with_pauses.append(pause_samples)
                # Remove the last added pause as it's not needed after the last segment
                audio_arr_segments = audio_with_pauses[:-1]

            # put all audio together
            try:
                audio_data_np_array = np.concatenate(audio_arr_segments)
            except ValueError as e:
                print(f"Failed to concatenate audio segments: {e}")
                return None

        else:
            text = prompt_wrap.replace("##", text)

            for retry_num in range(validate_max_retries):
                if not silent:
                    print(f"attempt: {retry_num}")
                if write_last_history_prompt:
                    if not use_vocos:
                        history_prompt_data, audio_data_np_array = self.bark_module.generate_audio(text,
                                                                                                   history_prompt=history_prompt,
                                                                                                   text_temp=text_temp,
                                                                                                   waveform_temp=waveform_temp,
                                                                                                   min_eos_p=min_eos_p,
                                                                                                   output_full=write_last_history_prompt,
                                                                                                   silent=silent,
                                                                                                   )
                    else:
                        # vocos_output = torchaudio.functional.resample(vocos_output, orig_freq=24000, new_freq=44100).cpu()
                        history_prompt_data, audio_data_np_array = self.generate_segments_vocos(text,
                                                                                                history_prompt=history_prompt,
                                                                                                text_temp=text_temp,
                                                                                                waveform_temp=waveform_temp,
                                                                                                min_eos_p=min_eos_p,
                                                                                                silent=silent,
                                                                                                )

                    self.bark_module.save_as_prompt(last_hisory_prompt_file, history_prompt_data)
                else:
                    if not use_vocos:
                        audio_data_np_array = self.bark_module.generate_audio(text,
                                                                              history_prompt=history_prompt,
                                                                              text_temp=text_temp,
                                                                              waveform_temp=waveform_temp,
                                                                              min_eos_p=min_eos_p,
                                                                              silent=silent,
                                                                              )
                    else:
                        _, audio_data_np_array = self.generate_segments_vocos(text,
                                                                              history_prompt=history_prompt,
                                                                              text_temp=text_temp,
                                                                              waveform_temp=waveform_temp,
                                                                              min_eos_p=min_eos_p,
                                                                              silent=silent,
                                                                              )

                audio_data_np_array = self.audio_processing(audio_data_np_array, skip_infinity_lufs=skip_infinity_lufs)

                if self.get_plugin_setting(
                        "validate_generated_audio") and audio_data_np_array is not None and audio_data_np_array.size > 0:
                    if not silent:
                        print(f"Validating generated audio for text...")
                    levenshtein_threshold = int(self.get_plugin_setting("validate_max_distance_threshold"))

                    generated_text = self.transcribe(audio_data_np_array)
                    if generated_text is not None and generated_text != "" and self._search_word_levenshtein(text, generated_text,
                                                                                                             levenshtein_threshold):
                        if not silent:
                            print("Generated audio is valid")
                        break
                    else:
                        print(f"Failed to generate non-empty audio for text after {validate_max_retries} retries")
                        continue  # Skip

                break

            audio_data_np_array = np.int16(audio_data_np_array * 32767)  # Convert to 16-bit PCM

            if self.get_plugin_setting("use_vocos_on_result"):
                if not silent:
                    print("applying vocos on result")
                audio_data_np_array = audio_tools.numpy_array_to_wav_bytes(audio_data_np_array, self.sample_rate)
                audio_data_np_array = self.apply_vocos_on_audio(audio_data_np_array)
                audio_data_np_array = audio_tools.wav_bytes_to_numpy_array(audio_data_np_array.getvalue())


        # if self.get_plugin_setting("audio_denoise", False):
        #    if self.audio_enhancer is None:
        #        self.audio_enhancer = DeepFilterNet.DeepFilterNet(post_filter=True)
        #    print("denoising audio")
        #    #audio_data_np_array = np.asarray(audio_data_np_array, dtype=np.float32)
        #    #audio_data_16bit = self.audio_enhancer.enhance_audio(audio_data_np_array, sample_rate=44100, output_sample_rate=self.sample_rate)
        #    audio_data_16bit = self.audio_enhancer.simple_enhance(audio_data_np_array, audio_sample_rate=self.sample_rate, output_sample_rate=self.sample_rate)
        # else:

        # if self.get_plugin_setting("audio_denoise", False):
        #    if self.audio_enhancer is None:
        #        self.audio_enhancer = DeepFilterNet.DeepFilterNet(post_filter=True)
        #    audio_data_np_array = self.audio_enhancer.simple_enhance(audio_data_np_array, audio_sample_rate=self.sample_rate, output_sample_rate=self.sample_rate)

        if audio_data_np_array is None:
            print("Audio generation was empty.")
            return None

        buff = io.BytesIO()
        write_wav(buff, self.sample_rate, audio_data_np_array)

        buff.seek(0)

        #if self.get_plugin_setting("use_vocos_on_result"):
        #    if not silent:
        #        print("applying vocos on result")
        #    buff = self.apply_vocos_on_audio(buff)

        # call custom plugin event method
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio',
                                                        {'audio': buff, 'sample_rate': self.sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            if not silent:
                print("applied plugin_tts_after_audio")
            buff = plugin_audio['audio']

        total_time_string = calculate_total_time(segment_times)

        if not silent:
            if not is_stopping:
                print(f"Audio generation finished. (100%){total_time_string}")
            else:
                print(f"Audio generation finished. (stopped){total_time_string}")

        return buff.getvalue()

    def timer(self):
        pass

    def play_audio_on_device(self, wav, audio_device, source_sample_rate=24000, audio_device_channel_num=2,
                             target_channels=2, dtype="int16"):
        secondary_audio_device = None
        if settings.GetOption("tts_use_secondary_playback") and (
                (settings.GetOption("tts_secondary_playback_device") == -1 and audio_device != settings.GetOption(
                    "device_default_out_index")) or
                (settings.GetOption("tts_secondary_playback_device") > -1 and audio_device != settings.GetOption(
                    "tts_secondary_playback_device"))):
            secondary_audio_device = settings.GetOption("tts_secondary_playback_device")
            if secondary_audio_device == -1:
                secondary_audio_device = settings.GetOption("device_default_out_index")

        audio_tools.play_audio(wav, audio_device,
                               source_sample_rate=source_sample_rate,
                               audio_device_channel_num=audio_device_channel_num,
                               target_channels=target_channels,
                               dtype=dtype,
                               secondary_device=secondary_audio_device, tag="tts")

    def stt(self, text, result_obj):
        if self.is_enabled(False) and settings.GetOption("tts_answer") and text.strip() != "":
            prompt_wrap = self.get_plugin_setting("prompt_wrap", "##")
            text_temp = self.get_plugin_setting("temperature_text")
            waveform_temp = self.get_plugin_setting("temperature_waveform")
            long_text = self.get_plugin_setting("long_text", False)
            long_text_stable_frequency = self.get_plugin_setting("long_text_stable_frequency")
            long_text_split_pause = self.get_plugin_setting("long_text_split_pause")
            write_last_history_prompt = self.get_plugin_setting("write_last_history_prompt", False)
            write_history_prompt_file = self.get_plugin_setting("write_last_history_prompt_file",
                                                                "bark_prompts/last_prompt.npz")
            if write_last_history_prompt:
                os.makedirs(os.path.dirname(write_history_prompt_file), exist_ok=True)

            audio_device = settings.GetOption("device_out_index")
            if audio_device is None or audio_device == -1:
                audio_device = settings.GetOption("device_default_out_index")
            wav = self.generate_tts(text.strip(),
                                    text_temp=text_temp,
                                    waveform_temp=waveform_temp,
                                    write_last_history_prompt=write_last_history_prompt,
                                    last_hisory_prompt_file=write_history_prompt_file,
                                    prompt_wrap=prompt_wrap,
                                    long_text=long_text, long_text_stable_frequency=long_text_stable_frequency,
                                    long_text_split_pause=long_text_split_pause, )
            if wav is not None:
                self.play_audio_on_device(wav, audio_device,
                                          source_sample_rate=self.sample_rate,
                                          audio_device_channel_num=2,
                                          target_channels=2,
                                          dtype="int16"
                                          )
        return

    def clone_voice(self):
        clone_voice_prompt = ""
        if self.get_plugin_setting("clone_voice_prompt", "") != "":
            clone_voice_prompt = self.get_plugin_setting("clone_voice_prompt", "")
        if clone_voice_prompt == "":
            websocket.BroadcastMessage(
                json.dumps({"type": "info", "data": "Please enter a prompt for the voice cloning."}))
            return

        websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Voice Cloning started..."}))
        clone_voice_audio_filepath = self.get_plugin_setting("clone_voice_audio_filepath",
                                                             "bark_clone_voice/clone_voice.wav")
        # get the directory of the clone audio file string
        os.makedirs(os.path.dirname(clone_voice_audio_filepath), exist_ok=True)
        clone_history_prompt_save = os.path.splitext(clone_voice_audio_filepath)[0]
        # check if clone_voice_audio_filepath is a file and exists
        if not os.path.isfile(clone_voice_audio_filepath):
            websocket.BroadcastMessage(json.dumps({"type": "info",
                                                   "data": "No clone audio file found. Please record a clone audio file first (between 4 - 8 seconds."}))
            return
        self.bark_module.clone_voice(
            audio_filepath=clone_voice_audio_filepath,
            text=clone_voice_prompt,
            dest_filename=clone_history_prompt_save,
        )
        websocket.BroadcastMessage(json.dumps(
            {"type": "info", "data": "Voice Cloning finished.\n\nLook in folder '" + clone_history_prompt_save + "'."}))

    def clone_voice_better(self):
        use_offload_cpu = self.get_plugin_setting("use_offload_cpu", True)
        use_gpu = self.get_plugin_setting("use_gpu", True)
        use_small_models = self.get_plugin_setting("use_small_models", True)

        websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Better Voice Cloning started..."}))

        if not Path(bark_plugin_dir / bark_voice_clone_tool["path"] / "barkVoiceClone.exe").is_file():
            # download from random url in list
            voice_clone_url = random.choice(bark_voice_clone_tool["urls"])
            downloader.download_extract([voice_clone_url],
                                        str(bark_plugin_dir.resolve()),
                                        bark_voice_clone_tool["sha256"],
                                        alt_fallback=True,
                                        fallback_extract_func=downloader.extract_zip,
                                        fallback_extract_func_args=(
                                            str(bark_plugin_dir / os.path.basename(voice_clone_url)),
                                            str(bark_plugin_dir.resolve()),
                                        ),
                                        title="Bark - voiceclone app", extract_format="zip")

        if Path(bark_plugin_dir / bark_voice_clone_tool["path"] / "barkVoiceClone.exe").is_file():
            clone_voice_audio_filepath = self.get_plugin_setting("clone_voice_audio_filepath",
                                                                 "bark_clone_voice/clone_voice.wav")
            os.makedirs(os.path.dirname(clone_voice_audio_filepath), exist_ok=True)
            clone_history_prompt_save = os.path.splitext(clone_voice_audio_filepath)[0] + ".npz"
            # run command line tool with parameters
            try:
                process_arguments = [str(bark_plugin_dir / bark_voice_clone_tool["path"] / "barkVoiceClone.exe"),
                                     "--audio_file", clone_voice_audio_filepath, "--npz_file",
                                     clone_history_prompt_save]
                if use_offload_cpu:
                    process_arguments.append("--offload_cpu")
                if use_gpu:
                    process_arguments.append("--use_gpu")
                if use_small_models:
                    process_arguments.append("--small_model")

                # add min_eos_p setting
                process_arguments.append("--min_eos_p")
                process_arguments.append(str(self.get_plugin_setting("min_eos_p")))

                subprocess.run(process_arguments, check=True)
            except subprocess.CalledProcessError as e:
                websocket.BroadcastMessage(
                    json.dumps({"type": "error", "data": "Better Voice Cloning failed: " + str(e)}))
                return

            websocket.BroadcastMessage(json.dumps({"type": "info",
                                                   "data": "Better Voice Cloning finished.\n\nLook in folder '" + clone_history_prompt_save + "'."}))
        else:
            websocket.BroadcastMessage(
                json.dumps({"type": "error", "data": "Better Voice Cloning failed: barkVoiceClone.exe not found."}))
        return

    def batch_generate(self):
        # generate multiple voices in a batch
        write_last_history_prompt = True
        prompt_wrap = "##"
        text_temp = self.get_plugin_setting("temperature_text")
        waveform_temp = self.get_plugin_setting("temperature_waveform")
        batch_prompts = self.get_plugin_setting("batch_prompts")
        batch_size = self.get_plugin_setting("batch_size")
        batch_folder = self.get_plugin_setting("batch_folder")
        os.makedirs(batch_folder, exist_ok=True)
        segment_times = []
        text_list = batch_prompts.split("\n")
        # remove empty lines
        text_list = [x for x in text_list if x.strip() != ""]

        if batch_size > 0:
            batch_total = batch_size * len(text_list)

            websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Batch Generating " + str(
                batch_total) + " audios...\n(" + str(
                batch_size) + " per prompt)\nstarted.\n\nlook for them in '" + batch_folder + "' directory."}))

            prompt_num = 0
            gen_num = 0
            for text_line in text_list:
                if self.stop_batch_processing:
                    self.stop_batch_processing = False
                    print("Batch generating stopped.")
                    websocket.BroadcastMessage(json.dumps({"type": "info",
                                                           "data": "Batch Generating stopped.\n\nlook for them in '" + batch_folder + "' directory."}))
                    return

                if text_line.strip() != "":
                    prmpt_dir = batch_folder + "/prompt-" + str(prompt_num)
                    os.makedirs(prmpt_dir, exist_ok=True)
                    # write prompt text to file
                    with open(prmpt_dir + "/prompt.txt", "w", encoding='utf-8') as f:
                        f.write(text_line.strip())
                    for i in range(batch_size):
                        if self.stop_batch_processing:
                            self.stop_batch_processing = False
                            print("Batch generating stopped.")
                            websocket.BroadcastMessage(json.dumps({"type": "info",
                                                                   "data": "Batch Generating stopped.\n\nlook for them in '" + batch_folder + "' directory."}))
                            return
                        start_time = time.time()
                        # Estimate time for the remaining segments after the first three are done
                        estimate_time_full_str = estimate_remaining_time(batch_total, segment_times, 3)

                        print(f"total batch progress: {int(gen_num / batch_total * 100)}% ({gen_num} of {batch_total}){estimate_time_full_str}")

                        file_name = prmpt_dir + "/" + str(i)
                        # generate wav and history prompt
                        wav = self.generate_tts(text_line.strip(),
                                                text_temp=text_temp,
                                                waveform_temp=waveform_temp,
                                                write_last_history_prompt=write_last_history_prompt,
                                                last_hisory_prompt_file=file_name + ".npz",
                                                prompt_wrap=prompt_wrap,
                                                skip_infinity_lufs=False, silent=True,)

                        # write wav to file
                        if wav is not None:
                            # write wav to file
                            wav_file_name = file_name + ".wav"
                            with open(wav_file_name, "wb") as f:
                                f.write(wav)
                        end_time = time.time()
                        segment_times.append(end_time - start_time)
                        gen_num += 1
                    prompt_num += 1

            total_time_string = calculate_total_time(segment_times)
            print(f"Batch Generating finished. (100%){total_time_string}")
            websocket.BroadcastMessage(json.dumps({"type": "info",
                                                   "data": "Batch Generating finished.\n\nlook for them in '" + batch_folder + "' directory."}))
        else:
            error_msg = "Invalid batch size. must be number of runs per line"
            print(error_msg)
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": error_msg}))

    def tts(self, text, device_index, websocket_connection=None, download=False):
        if self.is_enabled(False):
            prompt_wrap = self.get_plugin_setting("prompt_wrap", "##")
            text_temp = self.get_plugin_setting("temperature_text")
            waveform_temp = self.get_plugin_setting("temperature_waveform")
            long_text = self.get_plugin_setting("long_text", False)
            long_text_stable_frequency = self.get_plugin_setting("long_text_stable_frequency")
            long_text_split_pause = self.get_plugin_setting("long_text_split_pause")
            write_last_history_prompt = self.get_plugin_setting("write_last_history_prompt", False)
            last_hisory_prompt_file = self.get_plugin_setting("write_last_history_prompt_file",
                                                              "bark_prompts/last_prompt.npz")
            if write_last_history_prompt:
                os.makedirs(os.path.dirname(last_hisory_prompt_file), exist_ok=True)

            if device_index is None or device_index == -1:
                device_index = settings.GetOption("device_default_out_index")

            wav = self.generate_tts(text.strip(),
                                    text_temp=text_temp,
                                    waveform_temp=waveform_temp,
                                    write_last_history_prompt=write_last_history_prompt,
                                    last_hisory_prompt_file=last_hisory_prompt_file,
                                    prompt_wrap=prompt_wrap,
                                    long_text=long_text, long_text_stable_frequency=long_text_stable_frequency,
                                    long_text_split_pause=long_text_split_pause, )
            if wav is not None:
                if download and websocket_connection is not None:
                    wav_data = base64.b64encode(wav).decode('utf-8')
                    websocket.AnswerMessage(websocket_connection,
                                            json.dumps({"type": "tts_save", "wav_data": wav_data}))
                    del wav_data
                else:
                    self.play_audio_on_device(wav, device_index,
                                              source_sample_rate=self.sample_rate,
                                              audio_device_channel_num=2,
                                              target_channels=2,
                                              dtype="int16"
                                              )
                del wav
        return

    def on_event_received(self, message, websocket_connection=None):
        if self.is_enabled(False):
            if "type" not in message:
                return
            if message["type"] == "plugin_button_press":
                if message["value"] == "zz_clone_voice_better_button":
                    self.clone_voice_better()
                if message["value"] == "zz_clone_voice_button":
                    self.clone_voice()
                if message["value"] == "zz_batch_button":
                    self.batch_generate()
                if message["value"] == "zz_batch_stop_button":
                    print("Batch generating stopping...")
                    self.stop_batch_processing = True
                if message["value"] == "vocos_file_button":
                    if self.vocos is None and not self.get_plugin_setting("use_vocos"):
                        websocket.BroadcastMessage(json.dumps(
                            {"type": "info", "data": "Vocos is disabled. Please enable it in the settings."}))
                    elif self.vocos is None and self.get_plugin_setting("use_vocos"):
                        self._load_vocos_model()
                    wav_file = self.get_plugin_setting("vocos_file")
                    wav = self.apply_vocos_on_audio(wav_file).getvalue()

                    if wav is not None and websocket.UI_CONNECTED["websocket"] is not None:
                        wav_data = base64.b64encode(wav).decode('utf-8')
                        websocket.AnswerMessage(websocket.UI_CONNECTED["websocket"],
                                                json.dumps({"type": "tts_save", "wav_data": wav_data}))

        else:
            websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Plugin is disabled."}))

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        pass
