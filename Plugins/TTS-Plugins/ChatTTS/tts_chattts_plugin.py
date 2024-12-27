# ============================================================
# ChatTTS Text to Speech Plugin for Whispering Tiger
# V0.0.7
# ChatTTS: https://github.com/2noise/ChatTTS
# Whispering Tiger: https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import base64
import io
import json
import os
import random
import sys
import traceback
from importlib import util
import importlib
import pkgutil
from pathlib import Path

import numpy as np
import torch
import torchaudio

import Plugins
import audio_tools
import downloader

import settings
import websocket


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


plugin_dir = Path(Path.cwd() / "Plugins" / "chattts_plugin")
os.makedirs(plugin_dir, exist_ok=True)

chattts_dependency_module = {
    "url": "https://github.com/Sharrnah/ChatTTS/archive/refs/heads/0.1.1_fix-eng-chars.zip",
    "sha256": "67d683b140d802eba7db90f4fafcb9de52a91c674972e02560671fcc9d9afcf8",
    "path": "ChatTTS-0.1.1_fix-eng-chars/ChatTTS",
}
chattts_models = {
    "urls": [
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/chat-tts/chat_tts_models.zip",
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/chat-tts/chat_tts_models.zip",
    ],
    "file_checksums": {
        "asset/Decoder.pt": "9964e36e840f0e3a748c5f716fe6de6490d2135a5f5155f4a642d51860e2ec38",
        "asset/DVAE.pt": "613cb128adf89188c93ea5880ea0b798e66b1fe6186d0c535d99bcd87bfd6976",
        "asset/GPT.pt": "d7d4ee6461ea097a2be23eb40d73fb94ad3b3d39cb64fbb50cb3357fd466cadb",
        "asset/spk_stat.pt": "3228d8a4cbbf349d107a1b76d2f47820865bd3c9928c4bdfe1cefd5c7071105f",
        "asset/tokenizer.pt": "e911ae7c6a7c27953433f35c44227a67838fe229a1f428503bdb6cd3d1bcc69c",
        "asset/Vocos.pt": "09a670eda1c08b740013679c7a90ebb7f1a97646ea7673069a6838e6b51d6c58",
    },
    "sha256": "4bf2db1da0ceb1610e1d91e3edebe92b484fbb381e59d74464a8a8f936983678",
    "path": "models",
}

einx_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/90/04/4a730d74fd908daad86d6b313f235cdf8e0cf1c255b392b7174ff63ea81a/einx-0.3.0-py3-none-any.whl",
    "sha256": "367d62bab8dbb8c4937308512abb6f746cc0920990589892ba0d281356d39345",
    "path": "einx"
}

vocos_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/98/b3/445694d1059688a76a997c61936fef938b7d90f905a00754b4a441e7fcbd/vocos-0.0.3-py3-none-any.whl",
    "sha256": "0578b20b4ba57533a9d9b3e5ec3f81982f6fabd07ef02eb175fa9ee5da1e3cac",
    "path": "vocos"
}

vector_quantize_pytorch_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/77/5c/1e230ac0a4ef760d35ebaf83ee1fc63603fabd4921acfd16ca7037f8f605/vector_quantize_pytorch-1.15.2-py3-none-any.whl",
    "sha256": "8cd8ec731c3c378fb79e792f1cb3a79086bcca7f9150b9ddc9e1569a80cab1cc",
    "path": "vector_quantize_pytorch"
}
pybase16384_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/0d/ca/937f5e34af0f3007059b77da0242aae9e40507fbe2291c76e7e77c1ea6fb/pybase16384-0.3.7-cp311-cp311-win_amd64.whl",
    "sha256": "d059bb5b673ed08e249957b0dcbf263ae7fc6af2ecd415bc0207762d67794c9c",
    "path": "pybase16384"
}

encodec_dependency = {
    "url": "https://files.pythonhosted.org/packages/62/59/e47bbd0542d0e6f4ce9983d5eb458a01d4b42c81e5c410cb9e159b1061ae/encodec-0.1.1.tar.gz",
    "sha256": "36dde98ccfe6c51a15576476cadfcb3b35a63507b8b8555abd69889a6fba6772",
    "path": "encodec-0.1.1/encodec"
}


class ChatTTSPlugin(Plugins.Base):
    chattts_module = None

    model = None
    encodec = None
    vocos = None

    sample_rate = 24000

    speaker_dir = Path(plugin_dir / "speaker")

    def init(self):
        self.init_plugin_settings(
            {
                "speaker_file": {"type": "file_open", "accept": ".spk,.pt",
                                 "value": str(Path(self.speaker_dir / "my_speaker.spk").resolve())},
                "temperature": {"type": "slider", "min": 0.0001, "max": 1.0000, "step": 0.0001, "value": 0.0003},
                "top_p": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.1, "value": 0.7},
                "top_k": {"type": "slider", "min": 1, "max": 100, "step": 1, "value": 20},
                "repetition_penalty": {"type": "slider", "min": 0.00, "max": 2.00, "step": 0.01, "value": 1.05},
                "max_new_token": {"type": "slider", "min": 1, "max": 4000, "step": 1, "value": 2048},
                "min_new_token": {"type": "slider", "min": 0, "max": 1000, "step": 1, "value": 0},
                "speaker_file_convert_pt": {"label": "Convert .pt speaker file", "type": "button", "style": "default"},

                "prompt": "",
                "text_wrap": "#!# [uv_break]",
                "text_wrap_info": {
                    "label": "wraps text. #!# will be replaced with the text.",
                    "type": "label", "style": "left"},
                "seed": -1,
                "skip_refine_text": False,
                "do_text_normalization": True,
                "do_homophone_replacement": True,
                "language": {"type": "select", "value": "Auto", "values": ["Auto", "en", "zh"]},
                "device": {"type": "select", "value": "auto",
                           "values": [
                               "auto",
                               "cpu:0", "cpu:1", "cpu:2",
                               "cuda:0", "cuda:1", "cuda:2",
                               #"direct-ml:0", "direct-ml:1", "direct-ml:2"  # direct-ml does not seem to work
                           ]},
            },
            settings_groups={
                "General": ["speaker_file", "temperature", "top_p", "top_k", "repetition_penalty", "max_new_token",
                            "min_new_token", "speaker_file_convert_pt"],
                "Options": ["prompt", "text_wrap", "seed", "skip_refine_text", "do_text_normalization",
                            "do_homophone_replacement", "language", "device"],
            }
        )
        if self.is_enabled(False):
            print("loading ChatTTS...")
            # load the encodec module
            self.encodec = self._module_loader(encodec_dependency, "encodec module", extract_format="tar.gz")

            self._load_vocos_model()

            # load the einx module
            _ = self._module_loader(einx_dependency_module, "einx module", extract_format="zip")

            # load the vector_quantize_pytorch module
            _ = self._module_loader(vector_quantize_pytorch_dependency_module, "vector_quantize_pytorch module",
                                    extract_format="zip")

            # load the pybase16384 module
            _ = self._module_loader(pybase16384_dependency_module, "pybase16384 module", extract_format="zip")

            # load chattts models
            models_path = Path(plugin_dir / chattts_models["path"])
            if not downloader.check_file_hashes(str(models_path.resolve()), chattts_models["file_checksums"]):
                downloader.download_extract(chattts_models["urls"],
                                            str(models_path.resolve()),
                                            chattts_models["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(models_path / os.path.basename(
                                                    chattts_models["urls"][0])),
                                                str(models_path.resolve()),
                                            ),
                                            title="ChatTTS - Models", extract_format="zip")

            # load the chattts module
            self.chattts_module = self._module_loader(chattts_dependency_module, "main module", extract_format="zip")

            self.model = self.chattts_module.Chat()
            self.model.load(custom_path=str(Path(plugin_dir / "models").resolve()), compile=False, source="custom", device=self._get_infer_device())

            # disable default tts engine
            settings.SetOption("tts_type", "")

            print("ChatTTS loaded.")

            os.makedirs(self.speaker_dir, exist_ok=True)
        else:
            if self.model is not None:
                self.model.unload()
                del self.model
                print("ChatTTS unloaded.")

    def _get_infer_device(self):
        device_option = self.get_plugin_setting("device")
        device = device_option
        if isinstance(device_option, str) and (device_option == "cuda" or device_option == "auto"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device_option, str) and device_option.startswith("direct-ml"):
            device_id = 0
            device_id_split = device_option.split(":")
            if len(device_id_split) > 1:
                device_id = int(device_id_split[1])
            import torch_directml
            device = torch_directml.device(device_id)
        return device

    def _module_loader(self, module_dict, title="", extract_format="zip", recursive=False):
        fallback_extract_func = downloader.extract_zip
        if extract_format == "tar.gz":
            fallback_extract_func = downloader.extract_tar_gz

        module_path = Path(plugin_dir / module_dict["path"])
        if not Path(module_path / "__init__.py").is_file():
            downloader.download_extract([module_dict["url"]],
                                        str(plugin_dir.resolve()),
                                        module_dict["sha256"],
                                        alt_fallback=True,
                                        fallback_extract_func=fallback_extract_func,
                                        fallback_extract_func_args=(
                                            str(plugin_dir / os.path.basename(
                                                module_dict["url"])),
                                            str(plugin_dir.resolve()),
                                        ),
                                        title="ChatTTS - " + title, extract_format=extract_format)
        return load_module(str(module_path.resolve()), recursive=recursive)

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
        try:
            audio_tools.play_audio(wav, audio_device,
                                   source_sample_rate=source_sample_rate,
                                   audio_device_channel_num=audio_device_channel_num,
                                   target_channels=target_channels,
                                   dtype=dtype,
                                   secondary_device=secondary_audio_device, tag="tts")
        except Exception as e:
            print(f"Failed to play audio on device: {e}")
            traceback.print_exc()

    def _load_vocos_model(self):
        # load the vocos module (optional vocoder)
        if self.get_plugin_setting("use_vocos", True) and self.vocos is None:
            if not Path(plugin_dir / vocos_dependency_module["path"] / "__init__.py").is_file():
                downloader.download_extract([vocos_dependency_module["url"]],
                                            str(plugin_dir.resolve()),
                                            vocos_dependency_module["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(plugin_dir / os.path.basename(vocos_dependency_module["url"])),
                                                str(plugin_dir.resolve()),
                                            ),
                                            title="ChatTTS - vocos module", extract_format="zip")
            vocos_module = load_module(str(Path(plugin_dir / vocos_dependency_module["path"]).resolve()))
            device = self._get_infer_device()
            self.vocos = vocos_module.Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device)

    def generate_tts(self, text, speaker=None) -> torch.Tensor | None:
        temperature = self.get_plugin_setting("temperature", .3)
        top_p = self.get_plugin_setting("top_p", 0.7)
        top_k = self.get_plugin_setting("top_k", 20)
        repetition_penalty = self.get_plugin_setting("repetition_penalty", 1.0)
        max_new_token = self.get_plugin_setting("max_new_token", 384)
        min_new_token = self.get_plugin_setting("min_new_token", 0)
        prompt = self.get_plugin_setting("prompt", "[speed_5]")
        seed = int(self.get_plugin_setting("seed", -1))
        text_wrap = self.get_plugin_setting("text_wrap", "#!#")

        skip_refine_text = self.get_plugin_setting("skip_refine_text", False)
        do_text_normalization = self.get_plugin_setting("do_text_normalization", True)
        do_homophone_replacement = self.get_plugin_setting("do_homophone_replacement", True)
        language = self.get_plugin_setting("language", None)
        if language is not None and (language == "" or language.lower() == "auto"):
            language = None

        if seed <= -1:
            seed = random.randint(0, 2 ** 32 - 1)
        torch.manual_seed(seed)

        # wrap text
        if "#!#" in text_wrap:
            text = text_wrap.replace("#!#", text)

        params_refine_text = self.chattts_module.Chat.RefineTextParams(
            prompt=prompt,
        )

        params_infer_code = self.chattts_module.Chat.InferCodeParams(
            spk_emb=speaker,  # add sampled speaker
            temperature=temperature,  # using custom temperature
            top_P=top_p,  # top P decode
            top_K=top_k,  # top K decode
            repetition_penalty=repetition_penalty,
            max_new_token=max_new_token,
            min_new_token=min_new_token,
            #prompt=prompt,
        )

        wavs = self.model.infer(
            [text],
            lang=language,
            skip_refine_text=skip_refine_text,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,

            do_text_normalization=do_text_normalization,
            do_homophone_replacement=do_homophone_replacement,
        )

        #torchaudio.save("chattts_output1.wav", torch.from_numpy(wavs[0]), 24000)

        print(f"Finished (100%) [used seed: {seed}]")

        if wavs is None or len(wavs) <= 0:
            return None

        # convert to int16 and bytes
        wav_numpy = np.clip(wavs[0], -1.0, 1.0)

        buff = io.BytesIO()
        audio_data = np.int16(wav_numpy * 32767)

        torchaudio.save(buff, torch.from_numpy(audio_data), self.sample_rate, format="wav", encoding="PCM_S",
                        bits_per_sample=16)

        # call custom plugin event method
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio',
                                                        {'audio': buff.getvalue(), 'sample_rate': self.sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            return plugin_audio['audio']

        return buff.getvalue()

    def generate_random_speaker(self, speaker_file=None):
        speaker = self.model.sample_random_speaker()
        #write speaker to text file for testing
        speaker_file_path = str(Path(self.speaker_dir / speaker_file).resolve())
        self.save_speaker(speaker, speaker_file_path)
        return speaker

    def save_speaker(self, speaker_embedding_data: str | torch.Tensor | None = None, save_path: str | None = None):
        """
        Args:
            speaker_embedding_data: text data or torch tensor of speaker embedding data
            save_path: string of speaker file name to save to
        """
        speaker = None
        if speaker_embedding_data is not None:
            if isinstance(speaker_embedding_data, str):
                speaker = speaker_embedding_data
            elif isinstance(speaker_embedding_data, torch.Tensor):
                speaker = self.convert_speaker(speaker_embedding_data)
            else:
                print("Speaker embedding data of unknown format.")
        else:
            print("Speaker embedding data is required to save speaker.")

        if speaker is not None and save_path is not None and isinstance(save_path, str):
            with open(str(save_path), 'w', encoding='utf-8') as f:
                f.write(speaker)
                f.close()

    def convert_speaker(self, speaker_embedding: torch.Tensor | None = None):
        speaker = self.model._encode_spk_emb(speaker_embedding)
        return speaker

    def load_speaker(self, speaker_file=None):
        if Path(speaker_file).is_file():
            with open(str(speaker_file), 'r', encoding='utf-8') as f:
                speaker = f.read()
                f.close()
            return speaker
        else:
            print(f"Speaker file {speaker_file} not found.")
            return None

    def tts(self, text, device_index, websocket_connection=None, download=False, path=''):
        if self.is_enabled(False):
            if device_index is None or device_index == -1:
                device_index = settings.GetOption("device_default_out_index")

            speaker_file = self.get_plugin_setting("speaker_file")
            speaker = None
            if speaker_file is not None and isinstance(speaker_file, str) and speaker_file != "" and Path(
                    speaker_file).is_file():
                if Path(speaker_file).suffix == ".pt":
                    spk_tensor = torch.load(speaker_file, map_location=self._get_infer_device()).detach()
                    speaker = self.convert_speaker(spk_tensor)
                    if not Path(speaker_file).with_suffix(".spk").is_file():
                        self.save_speaker(spk_tensor, str(Path(speaker_file).with_suffix(".spk")))
                elif Path(speaker_file).suffix == ".spk":
                    speaker = self.load_speaker(speaker_file)
                else:
                    print(f"Invalid speaker file format. Expected '.pt' or '.spk'. Using random speaker.")
                    speaker = self.generate_random_speaker(f'random_speaker.spk')

            if speaker is None:
                print(f"No speaker found in {speaker_file}. Using random speaker.")
                speaker = self.generate_random_speaker(f'random_speaker.spk')

            wav_bytes = self.generate_tts(text.strip(), speaker=speaker)
            if wav_bytes is None:
                print("No audio to process.")
                return

            if download:
                if path is not None and path != '':
                    # write wav_data to file in path
                    with open(path, "wb") as f:
                        f.write(wav_bytes)
                    websocket.BroadcastMessage(json.dumps({"type": "info",
                                                           "data": "File saved to: " + path}))
                else:
                    if websocket_connection is not None:
                        wav_data = base64.b64encode(wav_bytes).decode('utf-8')
                        websocket.AnswerMessage(websocket_connection,
                                                json.dumps({"type": "tts_save", "wav_data": wav_data}))
            else:
                self.play_audio_on_device(wav_bytes, device_index,
                                          source_sample_rate=self.sample_rate,
                                          audio_device_channel_num=2,
                                          target_channels=2,
                                          dtype="int16"
                                          )
        return

    def stt(self, text, result_obj):
        if self.is_enabled(False) and settings.GetOption("tts_answer") and text.strip() != "":
            device_index = settings.GetOption("device_out_index")
            if device_index is None or device_index == -1:
                device_index = settings.GetOption("device_default_out_index")

            speaker_file = self.get_plugin_setting("speaker_file")
            if speaker_file is not None and isinstance(speaker_file, str) and speaker_file != "" and Path(
                    speaker_file).is_file():
                if Path(speaker_file).suffix == ".pt":
                    spk_tensor = torch.load(speaker_file, map_location=torch.device('cpu')).detach()
                    speaker = self.convert_speaker(spk_tensor)
                elif Path(speaker_file).suffix == ".spk":
                    speaker = self.load_speaker(speaker_file)
                else:
                    print(f"Invalid speaker file format. Expected '.pt' or '.spk'. Using random speaker.")
                    speaker = self.generate_random_speaker(f'random_speaker.spk')
            else:
                speaker = self.generate_random_speaker(f'random_speaker.spk')

            wav_bytes = self.generate_tts(text.strip(), speaker=speaker)
            if wav_bytes is None:
                print("No audio to process.")
                return

            self.play_audio_on_device(wav_bytes, device_index,
                                      source_sample_rate=self.sample_rate,
                                      audio_device_channel_num=2,
                                      target_channels=2,
                                      dtype="int16"
                                      )

    def on_event_received(self, message, websocket_connection=None):
        if self.is_enabled(False):
            if "type" not in message:
                return
            if message["type"] == "plugin_button_press":
                if message["value"] == "speaker_file_convert_pt":
                    speaker_file = self.get_plugin_setting("speaker_file")
                    if Path(speaker_file).suffix == ".pt":
                        spk_tensor = torch.load(speaker_file, map_location=torch.device('cpu')).detach()
                        if not Path(speaker_file).with_suffix(".spk").is_file():
                            self.save_speaker(spk_tensor, str(Path(speaker_file).with_suffix(".spk")))
                            websocket.BroadcastMessage(
                                json.dumps({"type": "info", "data": "Speaker saved as .spk file"}))
                        else:
                            websocket.BroadcastMessage(
                                json.dumps({"type": "info", "data": "Speaker already exists as .spk file"}))

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        self.init()
        pass
