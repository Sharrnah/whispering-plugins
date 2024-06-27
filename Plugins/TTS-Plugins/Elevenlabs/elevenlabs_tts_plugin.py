# ============================================================
# Elevenlabs TTS plugin for Whispering Tiger
# V1.0.11
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
import threading

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
from typing import BinaryIO, Union

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
    "url": "https://files.pythonhosted.org/packages/3c/4e/746741b1cdaf599de53651bb04457fe2aa53f264d6d369346879108b253b/elevenlabs-0.2.27-py3-none-any.whl",
    "sha256": "c31ea892d5668002bc26d0bb46a6466b0b4e2fe5aaed75cbc1b7011f01d3fa29",
    "path": "elevenlabs",
    "version": "0.2.27"
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
                "model_id": {"type": "select", "value": "eleven_multilingual_v1", "values": ["eleven_multilingual_v1", "eleven_multilingual_v2", "eleven_english_v2", "eleven_turbo_v2", "eleven_monolingual_v1"]},

                # Voice Settings
                "voice_stability": None,
                "voice_similarity_boost": None,
                "stt_min_words": 1,
                "stt_max_words": 40,
                "stt_max_char_length": 200,

                "streamed_playback": False,

                # Account
                "api_key": {"type": "textfield", "value": "", "password": True},
            },
            settings_groups={
                "General": ["model_id", "streamed_playback"],
                "Voice Settings": ["voice_stability", "voice_similarity_boost", "stt_min_words", "stt_max_words", "stt_max_char_length"],
                "Account": ["api_key"],
            }
        )

        if self.is_enabled(False):
            # load the elevenlabs module
            needs_update = should_update_version_file_check(
                Path(elevenlabs_plugin_dir / elevenlabs_dependency_module["path"]),
                elevenlabs_dependency_module["version"]
            )
            if needs_update and Path(elevenlabs_plugin_dir / elevenlabs_dependency_module["path"]).is_dir():
                print("Removing old elevenlabs directory")
                shutil.rmtree(str(Path(elevenlabs_plugin_dir / elevenlabs_dependency_module["path"]).resolve()))
            if not Path(elevenlabs_plugin_dir / elevenlabs_dependency_module["path"] / "__init__.py").is_file() or needs_update:
                downloader.download_extract([elevenlabs_dependency_module["url"]],
                                            str(elevenlabs_plugin_dir.resolve()),
                                            elevenlabs_dependency_module["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(elevenlabs_plugin_dir / os.path.basename(elevenlabs_dependency_module["url"])),
                                                str(elevenlabs_plugin_dir.resolve()),
                                            ),
                                            title="elevenlabs module", extract_format="zip")
                # write version file
                write_version_file(
                    Path(elevenlabs_plugin_dir / elevenlabs_dependency_module["path"]),
                    elevenlabs_dependency_module["version"]
                )

            self.elevenlabslib = load_module(
                str(Path(elevenlabs_plugin_dir / elevenlabs_dependency_module["path"]).resolve()))

            # disable default tts engine
            settings.SetOption("tts_enabled", False)

            threading.Thread(target=self._login).start()
        pass

    def _login(self):
        print("Logging in to Elevenlabs...")
        api_key = self.get_plugin_setting("api_key")
        os.environ["ELEVEN_API_KEY"] = api_key
        if api_key is None or api_key == "":
            print("No API key set or login failed")
            return
        self.voices = self.elevenlabslib.voices()

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
        voice_index = self.get_plugin_setting("voice_index", 0)
        model_id = self.get_plugin_setting("model_id", "eleven_multilingual_v1")
        stability = self.get_plugin_setting("voice_stability", None)
        similarity_boost = self.get_plugin_setting("voice_similarity_boost", None)

        if voice_name is None or voice_name == "" or self.elevenlabslib is None:
            print("No API instance or voice name set")
            return

        try:
            selected_voice = self._get_voices_by_name(voice_name)

            voice_settings = selected_voice.fetch_settings()
            if stability is not None:
                voice_settings.stability = float(stability)
            if similarity_boost is not None:
                voice_settings.similarity_boost = float(similarity_boost)

            audio_data = self.elevenlabslib.generate(text=text.strip(),
                                                     voice=self.elevenlabslib.Voice(
                                                         voice_id=selected_voice.voice_id,
                                                         settings=voice_settings
                                                     ),
                                                     model=model_id,
                                                     )

            # convert TTS to wav
            raw_data = io.BytesIO()
            save_audio_bytes(audio_data, raw_data, "wav")

            # call custom plugin event method
            plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': raw_data, 'sample_rate': self.source_sample_rate})
            if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
                raw_data = plugin_audio['audio']

            return raw_data.getvalue()

        except Exception as e:
            print(e)

    def generate_tts_streamed(self, text):
        if len(text.strip()) == 0:
            return None
        #voice_name = self.get_plugin_setting("voice", "Bella")
        voice_name = settings.GetOption("tts_voice")
        voice_index = self.get_plugin_setting("voice_index", 0)
        model_id = self.get_plugin_setting("model_id", "eleven_multilingual_v1")
        stability = self.get_plugin_setting("voice_stability", None)
        similarity_boost = self.get_plugin_setting("voice_similarity_boost", None)

        #if self.client is None or voice_name is None:
        if voice_name is None or voice_name == "" or self.elevenlabslib is None:
            print("No API instance or voice name set")
            return

        self.init_audio_stream_playback()
        try:
            selected_voice = self._get_voices_by_name(voice_name)

            voice_settings = selected_voice.fetch_settings()
            if stability is not None:
                voice_settings.stability = float(stability)
            if similarity_boost is not None:
                voice_settings.similarity_boost = float(similarity_boost)

            audio_data_stream = self.elevenlabslib.generate(text=text.strip(),
                                               voice=self.elevenlabslib.Voice(
                                                   voice_id=selected_voice.voice_id,
                                                   settings=voice_settings
                                               ),
                                               model=model_id,
                                               output_format="pcm_24000",
                                               latency=1,
                                               stream=True,
                                               stream_chunk_size=2048,
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


    def play_audio_on_device(self, wav, audio_device, source_sample_rate=24000, audio_device_channel_num=2, target_channels=2, input_channels=1, dtype="int16"):
        secondary_audio_device = None
        if settings.GetOption("tts_use_secondary_playback") and (
                (settings.GetOption("tts_secondary_playback_device") == -1 and audio_device != settings.GetOption("device_default_out_index")) or
                (settings.GetOption("tts_secondary_playback_device") > -1 and audio_device != settings.GetOption("tts_secondary_playback_device"))):
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

    def tts(self, text, device_index, websocket_connection=None, download=False):
        streamed_playback = self.get_plugin_setting("streamed_playback")

        if self.is_enabled(False):
            if device_index is None or device_index == -1:
                device_index = settings.GetOption("device_default_out_index")

            if not streamed_playback:
                wav = self.generate_tts(text.strip())
                if wav is not None:
                    if download and websocket_connection is not None:
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
                if download and websocket_connection is not None:
                    wav = self.generate_tts(text.strip())
                    if wav is not None:
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
