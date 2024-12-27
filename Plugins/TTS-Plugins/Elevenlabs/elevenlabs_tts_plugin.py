# ============================================================
# Elevenlabs TTS plugin for Whispering Tiger
# V1.1.3
#
# See https://github.com/Sharrnah/whispering-ui
# Uses the TTS engine from https://www.elevenlabs.com/
# ============================================================
import base64
import io
import json
import os
import re
import shutil

import numpy as np

import Plugins
import settings
import audio_tools

from pathlib import Path
import sys
from importlib import util
import downloader

import soundfile
import soundfile as sf
from scipy.io.wavfile import write as write_wav
from typing import BinaryIO, Union, Iterator
import websocket


def load_module(package_dir):
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

    # Remove the parent directory from sys.path
    sys.path.pop(0)

    return module


elevenlabs_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/48/43/77c7266f50b2e2eca5b155f15a35ef2a30cbd377d02a7f3ca32c077cb072/elevenlabs-1.50.3-py3-none-any.whl",
    "sha256": "13622d27f5ccd4c8bc793abbc252d43cdddc261805f378ae9a032a6458687589",
    "path": "elevenlabs",
    "version": "1.50.3"
}
httpx_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/41/7b/ddacf6dcebb42466abd03f368782142baa82e08fc0c1f8eaa05b4bae87d5/httpx-0.27.0-py3-none-any.whl",
    "sha256": "71d5465162c13681bff01ad59b2cc68dd838ea1f10e51574bac27103f00c91a5",
    "path": "httpx",
    "version": "0.27.0"
}
sniffio_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/e9/44/75a9c9421471a6c4805dbf2356f7c181a29c1879239abab1ea2cc8f38b40/sniffio-1.3.1-py3-none-any.whl",
    "sha256": "2f6da418d1f1e0fddd844478f41680e794e6051915791a034ff65e5f100525a2",
    "path": "sniffio",
    "version": "1.3.1"
}
httpcore_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/78/d4/e5d7e4f2174f8a4d63c8897d79eb8fe2503f7ecc03282fee1fa2719c2704/httpcore-1.0.5-py3-none-any.whl",
    "sha256": "421f18bac248b25d310f3cacd198d55b8e6125c107797b609ff9b7a6ba7991b5",
    "path": "httpcore",
    "version": "1.0.5"
}
h11_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/95/04/ff642e65ad6b90db43e668d70ffb6736436c7ce41fcc549f4e9472234127/h11-0.14.0-py3-none-any.whl",
    "sha256": "e3fe4ac4b851c468cc8363d500db52c2ead036020723024a109d37346efaa761",
    "path": "h11",
    "version": "0.14.0"
}

elevenlabs_plugin_dir = Path(Path.cwd() / "Plugins" / "elevenlabs_plugin")
os.makedirs(elevenlabs_plugin_dir, exist_ok=True)


def should_update_version_file_check(directory, current_version):
    # check version from VERSION file
    version_file = Path(directory / "WT_VERSION")
    if version_file.is_file():
        version = version_file.read_text().strip()
        if version != current_version:
            return True
        else:
            return False
    return True


def write_version_file(directory, version):
    version_file = Path(directory / "WT_VERSION")
    version_file.write_text(version)


class ElevenlabsTTSPlugin(Plugins.Base):
    httpx_lib = None
    elevenlabslib = None
    client = None
    voices = []
    # audio options
    source_dtype = "int16"
    source_sample_rate = 44100
    source_sample_rate_stream = 24000
    input_channels = 1
    target_channels = 2

    audio_streamer = None

    latency_optimizations = {
        "default (no latency optimizations)": 0,
        "normal (about 50% of max)": 1,
        "strong (about 75% of max)": 2,
        "max": 3,
        "max with text normalizer turned off (best latency)": 4
    }

    # return list of keys from latency_optimizations
    def get_latency_optimizations_list(self):
        return list(self.latency_optimizations.keys())

    # return value of provided latency_optimization key
    def get_latency_optimization_value(self, key):
        return self.latency_optimizations.get(key, 0)

    def _load_python_module(self, dependency_module, module_name="module"):
        # load the elevenlabs module
        needs_update = should_update_version_file_check(
            Path(elevenlabs_plugin_dir / dependency_module["path"]),
            dependency_module["version"]
        )
        if needs_update and Path(elevenlabs_plugin_dir / dependency_module["path"]).is_dir():
            print(f"Removing old {module_name} directory")
            shutil.rmtree(str(Path(elevenlabs_plugin_dir / dependency_module["path"]).resolve()))
        if not Path(elevenlabs_plugin_dir / dependency_module[
            "path"] / "__init__.py").is_file() or needs_update:
            downloader.download_extract([dependency_module["url"]],
                                        str(elevenlabs_plugin_dir.resolve()),
                                        dependency_module["sha256"],
                                        alt_fallback=True,
                                        fallback_extract_func=downloader.extract_zip,
                                        fallback_extract_func_args=(
                                            str(elevenlabs_plugin_dir / os.path.basename(
                                                dependency_module["url"])),
                                            str(elevenlabs_plugin_dir.resolve()),
                                        ),
                                        title=module_name, extract_format="zip")
            # write version file
            write_version_file(
                Path(elevenlabs_plugin_dir / dependency_module["path"]),
                dependency_module["version"]
            )
        return load_module(str(Path(elevenlabs_plugin_dir / dependency_module["path"]).resolve()))


    def word_char_count_allowed(self, text):
        word_count = len(re.findall(r'\w+', text))
        if self.get_plugin_setting("stt_min_words", 1) <= word_count <= self.get_plugin_setting("stt_max_words",
                                                                                                40) and self.get_plugin_setting(
            "stt_max_char_length", 200) >= len(text):
            return True
        else:
            return False

    def numpy_array_to_wav_bytes(self, audio: np.ndarray, sample_rate: int = 22050) -> io.BytesIO:
        buff = io.BytesIO()
        write_wav(buff, sample_rate, audio)
        buff.seek(0)
        return buff

    def get_plugin(self, class_name):
        for plugin_inst in Plugins.plugins:
            if plugin_inst.__class__.__name__ == class_name:
                return plugin_inst  # return plugin instance
        return None

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                # General
                #"voice_index": 0,
                "model_id": {"type": "select", "value": "eleven_multilingual_v2",
                             "values": ["eleven_multilingual_v1", "eleven_multilingual_v2", "eleven_english_v2",
                                        "eleven_turbo_v2", "eleven_turbo_v2_5", "eleven_flash_v2", "eleven_flash_v2_5", "eleven_monolingual_v1"]},

                # Voice Settings
                "voice_stability": {"type": "slider", "min": -0.01, "max": 1.00, "step": 0.01, "value": 0.71},
                "voice_similarity_boost": {"type": "slider", "min": -0.01, "max": 1.00, "step": 0.01, "value": 0.50},
                "voice_style": {"type": "slider", "min": -0.01, "max": 1.00, "step": 0.01, "value": 0.00},
                "stt_min_words": 1,
                "stt_max_words": 40,
                "stt_max_char_length": 200,

                "streamed_playback": False,
                "optimize_streaming_latency": {"type": "select", "value": self.get_latency_optimizations_list()[0],
                                               "values": self.get_latency_optimizations_list()},

                # Account
                "api_key": {"type": "textfield", "value": "", "password": True},
            },
            settings_groups={
                "General": ["model_id", "streamed_playback", "optimize_streaming_latency"],
                "Voice Settings": ["voice_stability", "voice_similarity_boost", "voice_style", "stt_min_words", "stt_max_words",
                                   "stt_max_char_length"],
                "Account": ["api_key"],
            }
        )

        if self.is_enabled(False):
            # load the sniffio module (required by httpx)
            _ = self._load_python_module(sniffio_dependency_module, "sniffio module")
            # load the h11 module (required by httpcore)
            _ = self._load_python_module(h11_dependency_module, "h11 module")
            # load the httpcore module (required by httpx)
            _ = self._load_python_module(httpcore_dependency_module, "httpcore module")

            # load the httpx module (required by elevenlabs)
            self.httpx_lib = self._load_python_module(httpx_dependency_module, "httpx module")

            # load the elevenlabs module
            self.elevenlabslib = self._load_python_module(elevenlabs_dependency_module, "elevenlabs module")
            sys.path.append(str(elevenlabs_plugin_dir.resolve()))

            # disable default tts engine
            settings.SetOption("tts_type", "")

            self._login()
        pass

    def _login(self):
        print("Logging in to Elevenlabs...")
        api_key = self.get_plugin_setting("api_key")
        #os.environ["ELEVEN_API_KEY"] = api_key
        if api_key is None or api_key == "":
            print("No API key set or login failed")
            return

        from elevenlabs.client import ElevenLabs

        self.client = ElevenLabs(
            api_key=api_key
        )

        voices_response = self.client.voices.get_all()
        self.voices = voices_response.voices

        print("Logged in to Elevenlabs.")

        websocket.BroadcastMessage(json.dumps({
            "type": "available_tts_voices",
            "data": self._get_speaker_names(self.voices)
        }))

    def _get_speaker_names(self, speakers):
        """Get a list of formatted strings combining speaker names with style names."""
        style_names = []
        if speakers is None or speakers == []:
            print("No Voices found")
            return []
        for speaker in speakers:
            style_names.append(f"{speaker.name}")
        return style_names

    def _get_voices_by_name(self, name):
        if self.voices is None or self.voices == []:
            print("No Voices found")
            return
        for voice in self.voices:
            if voice.name == name:
                return voice
        return None

    def generate_tts(self, text):
        if len(text.strip()) == 0:
            return None
        voice_name = settings.GetOption("tts_voice")
        model_id = self.get_plugin_setting("model_id", "eleven_multilingual_v1")
        stability = self.get_plugin_setting("voice_stability", None)
        similarity_boost = self.get_plugin_setting("voice_similarity_boost", None)
        style = self.get_plugin_setting("voice_style", None)

        if voice_name is None or voice_name == "" or self.elevenlabslib is None or self.client is None:
            print("No API instance or voice name set")
            return

        try:
            selected_voice = self._get_voices_by_name(voice_name)

            if selected_voice.settings is None:
                settings_stability = 0.71
                settings_similarity_boost = 0.5
                settings_style = 0.0
                if stability is not None and float(stability) > -0.01:
                    settings_stability = float(stability)
                if similarity_boost is not None and float(similarity_boost) > -0.01:
                    settings_similarity_boost = float(similarity_boost)
                if style is not None and float(style) > -0.01:
                    settings_style = float(style)

                voice_settings = self.elevenlabslib.VoiceSettings(
                    stability=settings_stability, similarity_boost=settings_similarity_boost, style=settings_style, use_speaker_boost=True
                )
            else:
                voice_settings = selected_voice.settings

            audio_data = self.client.generate(text=text.strip(),
                                              voice=self.elevenlabslib.Voice(
                                                  voice_id=selected_voice.voice_id,
                                                  settings=voice_settings
                                              ),
                                              output_format="mp3_44100_128",
                                              model=model_id,
                                              )

            if isinstance(audio_data, Iterator):
                audio_data = b"".join(audio_data)

            # convert TTS to wav
            raw_data = io.BytesIO()
            save_audio_bytes(audio_data, raw_data, "wav")

            # call custom plugin event method
            plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio',
                                                            {'audio': raw_data, 'sample_rate': self.source_sample_rate})
            if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
                raw_data = plugin_audio['audio']

            return raw_data.getvalue()

        except Exception as e:
            print(e)

    def generate_tts_streamed(self, text):
        if len(text.strip()) == 0:
            return None
        voice_name = settings.GetOption("tts_voice")
        model_id = self.get_plugin_setting("model_id", "eleven_multilingual_v1")
        stability = self.get_plugin_setting("voice_stability", None)
        similarity_boost = self.get_plugin_setting("voice_similarity_boost", None)
        style = self.get_plugin_setting("voice_style", None)
        optimize_streaming_latency = self.get_latency_optimization_value(self.get_plugin_setting("optimize_streaming_latency"))

        if voice_name is None or voice_name == "" or self.elevenlabslib is None or self.client is None:
            print("No API instance or voice name set")
            return

        self.init_audio_stream_playback()
        try:
            selected_voice = self._get_voices_by_name(voice_name)

            if selected_voice.settings is None:
                settings_stability = 0.71
                settings_similarity_boost = 0.5
                settings_style = 0.0
                if stability is not None and float(stability) > -0.01:
                    settings_stability = float(stability)
                if similarity_boost is not None and float(similarity_boost) > -0.01:
                    settings_similarity_boost = float(similarity_boost)
                if style is not None and float(style) > -0.01:
                    settings_style = float(style)

                voice_settings = self.elevenlabslib.VoiceSettings(
                    stability=settings_stability, similarity_boost=settings_similarity_boost, style=settings_style, use_speaker_boost=True
                )
            else:
                voice_settings = selected_voice.settings

            audio_data_stream = self.client.generate(text=text.strip(),
                                                     voice=self.elevenlabslib.Voice(
                                                         voice_id=selected_voice.voice_id,
                                                         settings=voice_settings
                                                     ),
                                                     model=model_id,
                                                     output_format="pcm_24000",
                                                     stream=True,
                                                     optimize_streaming_latency=optimize_streaming_latency,
                                                     )
            # Iterate over the audio chunks
            for chunk in audio_data_stream:
                self.audio_streamer.add_audio_chunk(chunk)

        except Exception as e:
            print(e)

    def timer(self):
        pass

    def init_audio_stream_playback(self):
        audio_device = settings.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.GetOption("device_default_out_index")

        #if self.audio_streamer is not None:
        #    self.audio_streamer.stop()
        #    self.audio_streamer = None
        #else:
        if self.audio_streamer is None:
            self.audio_streamer = audio_tools.AudioStreamer(audio_device,
                                                            source_sample_rate=self.source_sample_rate_stream,
                                                            playback_channels=2,
                                                            buffer_size=2048,
                                                            input_channels=1,
                                                            dtype="int16",
                                                            tag="tts",
                                                            )

        #def before_playback_hook(data, sample_rate):
        #    # call custom plugin event method
        #    plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': data, 'sample_rate': sample_rate})
        #    if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
        #        return plugin_audio['audio']
        #self.audio_streamer.set_before_playback_hook(before_playback_hook)

        #self.audio_streamer.start_playback()

    def play_audio_on_device(self, wav, audio_device, source_sample_rate=24000, audio_device_channel_num=2,
                             target_channels=2, input_channels=1, dtype="int16"):
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
                               input_channels=input_channels,
                               dtype=dtype,
                               secondary_device=secondary_audio_device, tag="tts")

    def stt(self, text, result_obj):
        streamed_playback = self.get_plugin_setting("streamed_playback")

        if self.is_enabled(False) and settings.GetOption("tts_answer") and text.strip() != "":
            audio_device = settings.GetOption("device_out_index")
            if audio_device is None or audio_device == -1:
                audio_device = settings.GetOption("device_default_out_index")

            if self.word_char_count_allowed(text.strip()):
                if not streamed_playback:
                    wav = self.generate_tts(text.strip())
                    if wav is not None:
                        self.play_audio_on_device(wav, audio_device,
                                                  source_sample_rate=self.source_sample_rate,
                                                  audio_device_channel_num=self.target_channels,
                                                  target_channels=self.target_channels,
                                                  input_channels=self.input_channels,
                                                  dtype=self.source_dtype
                                                  )
                else:
                    self.generate_tts_streamed(text.strip())
        return

    def tts(self, text, device_index, websocket_connection=None, download=False, path=''):
        streamed_playback = self.get_plugin_setting("streamed_playback")

        if self.is_enabled(False):
            if device_index is None or device_index == -1:
                device_index = settings.GetOption("device_default_out_index")

            if not streamed_playback:
                wav = self.generate_tts(text.strip())
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
                                                  source_sample_rate=self.source_sample_rate,
                                                  audio_device_channel_num=self.target_channels,
                                                  target_channels=self.target_channels,
                                                  input_channels=self.input_channels,
                                                  dtype=self.source_dtype,
                                                  )
            else:
                if download:
                    wav = self.generate_tts(text.strip())
                    if wav is not None:
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
                    self.generate_tts_streamed(text.strip())

        return

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        if self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_streamer = None
        pass


## elevenlabs lib helper functions
def save_audio_bytes(audioData: bytes, saveLocation: Union[BinaryIO, str], outputFormat) -> None:
    """
        This function saves the audio data to the specified location OR file-like object.
        soundfile is used for the conversion, so it supports any format it does.

        Parameters:
            audioData: The audio data.
            saveLocation: The path (or file-like object) where the data will be saved.
            outputFormat: The format in which the audio will be saved
        """
    tempSoundFile = soundfile.SoundFile(io.BytesIO(audioData))

    if isinstance(saveLocation, str):
        with open(saveLocation, "wb") as fp:
            sf.write(fp, tempSoundFile.read(), tempSoundFile.samplerate, format=outputFormat)
    else:
        sf.write(saveLocation, tempSoundFile.read(), tempSoundFile.samplerate, format=outputFormat)
        if callable(getattr(saveLocation, "flush")):
            saveLocation.flush()
