# ============================================================
# Mars5 Text to Speech Plugin for Whispering Tiger
# V0.0.1
# Mars5: https://github.com/Camb-ai/MARS5-TTS
# Whispering Tiger: https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import base64
import io
import json
import os
import sys
from importlib import util
import importlib
import pkgutil
from pathlib import Path

import librosa
import torch

import Plugins
import audio_tools
import downloader

from scipy.io.wavfile import write as write_wav

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


plugin_dir = Path(Path.cwd() / "Plugins" / "mars5_plugin")
os.makedirs(plugin_dir, exist_ok=True)

mars5_dependency_module = {
    "url": "https://github.com/Camb-ai/MARS5-TTS/archive/7bf65bd705af674f49ae9b9d9fe2975c4fe5cbf7.zip",
    "sha256": "a114e65a81d23c4fda7277ab4bcbfc2dadd880b25683d201e60b36aeb6ad71bd",
    "path": "MARS5-TTS-7bf65bd705af674f49ae9b9d9fe2975c4fe5cbf7",
}

vocos_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/98/b3/445694d1059688a76a997c61936fef938b7d90f905a00754b4a441e7fcbd/vocos-0.0.3-py3-none-any.whl",
    "sha256": "0578b20b4ba57533a9d9b3e5ec3f81982f6fabd07ef02eb175fa9ee5da1e3cac",
    "path": "vocos"
}

encodec_dependency = {
    "url": "https://files.pythonhosted.org/packages/62/59/e47bbd0542d0e6f4ce9983d5eb458a01d4b42c81e5c410cb9e159b1061ae/encodec-0.1.1.tar.gz",
    "sha256": "36dde98ccfe6c51a15576476cadfcb3b35a63507b8b8555abd69889a6fba6772",
    "path": "encodec-0.1.1/encodec"
}

class Mars5TTSPlugin(Plugins.Base):
    model = None
    config_class = None
    encodec = None
    vocos = None

    sample_rate = 16000

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "reference_audio": {"type": "file_open", "accept": ".wav", "value": ""},
                "reference_transcript": ""
            },
            settings_groups={
                "General": ["reference_audio", "reference_transcript"],
            }
        )

        if self.is_enabled(False):
            #
            # # load the bark module
            # if not Path(plugin_dir / mars5_dependency_module["path"] / "inference.py").is_file():
            #     downloader.download_extract([mars5_dependency_module["url"]],
            #                                 str(plugin_dir.resolve()),
            #                                 mars5_dependency_module["sha256"],
            #                                 alt_fallback=True,
            #                                 fallback_extract_func=downloader.extract_zip,
            #                                 fallback_extract_func_args=(
            #                                     str(plugin_dir / os.path.basename(mars5_dependency_module["url"])),
            #                                     str(plugin_dir.resolve()),
            #                                 ),
            #                                 title="Mars5 - module", extract_format="zip")
            #
            # self.mars5_module = load_module(
            #     str(Path(plugin_dir / mars5_dependency_module["path"] / "inference.py").resolve()))

            # set cache path
            torch.hub.set_dir(str(Path(plugin_dir).resolve()))

            # load the encodec module
            encodec_path = Path(plugin_dir / encodec_dependency["path"])
            if not Path(encodec_path / "__init__.py").is_file():
                downloader.download_extract([encodec_dependency["url"]],
                                            str(plugin_dir.resolve()),
                                            encodec_dependency["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_tar_gz,
                                            fallback_extract_func_args=(
                                                str(plugin_dir / os.path.basename(encodec_dependency["url"])),
                                                str(plugin_dir.resolve()),
                                            ),
                                            title="Mars5 - encodec module", extract_format="tar.gz")
            self.encodec = load_module(str(Path(plugin_dir / encodec_dependency["path"]).resolve()))

            self._load_vocos_model()

            self.model, self.config_class = torch.hub.load(trust_repo=True, skip_validation=True,
                                                           source='github',
                                                           repo_or_dir='Camb-ai/mars5-tts',
                                                           model='mars5_english')

            self.sample_rate = self.model.sr


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
                                            title="Bark - vocos module", extract_format="zip")
            vocos_module = load_module(str(Path(plugin_dir / vocos_dependency_module["path"]).resolve()))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.vocos = vocos_module.Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device)

            # if self.model is not None and self.model.Vocos is None:
            #     self.model.Vocos = vocos_module

    def generate_tts(self, text, reference_wav=None, reference_transcript='', deep_clone=False):
        wav, sr = librosa.load(reference_wav,
                               sr=self.model.sr, mono=True)
        wav = torch.from_numpy(wav)
        print("Reference audio:")

        cfg = self.config_class(deep_clone=deep_clone, rep_penalty_window=100,
                                top_k=100, temperature=0.7, freq_penalty=3)

        ar_codes, wav_out = self.model.tts(text, wav,
                                           reference_transcript,
                                           cfg=cfg)

        buff = io.BytesIO()
        write_wav(buff, self.model.sr, wav_out.numpy())
        buff.seek(0)

        return buff.getvalue()

    def tts(self, text, device_index, websocket_connection=None, download=False, path=''):
        if self.is_enabled(False):
            if device_index is None or device_index == -1:
                device_index = settings.GetOption("device_default_out_index")

            deep_clone = False
            reference_transcript = self.get_plugin_setting("reference_transcript")
            if reference_transcript != "":
                deep_clone = True

            wav = self.generate_tts(text.strip(),
                                    reference_wav=self.get_plugin_setting("reference_audio"),
                                    reference_transcript=reference_transcript, deep_clone=deep_clone)
            if wav is not None:
                if download:
                    if path is not None and path != '':
                        # write wav_data to file in path
                        with open(path, "wb") as f:
                            f.write(wav)
                        websocket.BroadcastMessage(json.dumps({"type": "info",
                                                               "data": "File saved to: " + path}))
                    else:
                        if websocket_connection is not None:
                            wav_data = base64.b64encode(wav).decode('utf-8')
                            websocket.AnswerMessage(websocket_connection,
                                                    json.dumps({"type": "tts_save", "wav_data": wav_data}))
                else:
                    self.play_audio_on_device(wav, device_index,
                                              source_sample_rate=self.sample_rate,
                                              audio_device_channel_num=2,
                                              target_channels=2,
                                              dtype="int16"
                                              )
        return

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        pass
