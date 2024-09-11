# ============================================================
# Deepgram API - Whispering Tiger Plugin
# Version 0.0.2
# See https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import base64
import io
import json
import re
from typing import Union, BinaryIO

import requests
import soundfile

import Plugins
import audio_tools
import settings

import websocket

from whisper.tokenizer import LANGUAGES

LLM_LANGUAGES = [
]

TTS_VOICES = [
    "aura-asteria-en",
    "aura-luna-en",
    "aura-stella-en",
    "aura-athena-en",
    "aura-hera-en",
    "aura-orion-en",
    "aura-arcas-en",
    "aura-perseus-en",
    "aura-angus-en",
    "aura-orpheus-en",
    "aura-helios-en",
    "aura-zeus-en",
]


class DeepgramAPIPlugin(Plugins.Base):
    tts_source_sample_rate = 24000
    tts_source_dtype = "int16"
    tts_input_channels = 1
    tts_target_channels = 2

    audio_streamer = None

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "api_key": {"type": "textfield", "value": "", "password": True},

                # transcribe audio settings
                "transcribe_audio_enabled": False,
                "audio_transcribe_api_endpoint": "https://api.deepgram.com/v1/listen",
                "audio_model": {"type": "select", "value": "nova-2-general",
                                "values": [
                                    'nova-2-general',
                                    'nova-2-meeting',
                                    'nova-2-phonecall',
                                    'nova-2-finance',
                                    'nova-2-conversationalai',
                                    'nova-2-voicemail',
                                    'nova-2-video',
                                    'nova-2-medical',
                                    'nova-2-drivethru',
                                    'nova-2-atc',
                                    'nova-general',
                                    'nova-phonecall',
                                    'nova-medical',
                                    'enhanced-general',
                                    'enhanced-meeting',
                                    'enhanced-phonecall',
                                    'enhanced-finance',
                                    'base-general',
                                    'base-meeting',
                                    'base-phonecall',
                                    'base-finance',
                                    'base-conversationalai',
                                    'base-voicemail',
                                    'base-video',
                                    'whisper-tiny',
                                    'whisper-base',
                                    'whisper-small',
                                    'whisper-medium',
                                    'whisper-large',
                                ]},

                # text-to-speech settings
                "tts_enabled": False,
                "tts_api_endpoint": "https://api.deepgram.com/v1/speak",
                "tts_voice": {"type": "select", "value": TTS_VOICES[0],
                              "values": TTS_VOICES},
                "stt_min_words": -1,
                "stt_max_words": -1,
                "stt_max_char_length": -1,
                "streamed_playback": False
            },
            settings_groups={
                "General": ["api_key", "translate_enabled", "transcribe_audio_enabled", "tts_enabled"],
                "Speech-to-Text": ["audio_transcribe_api_endpoint", "audio_model"],
                "Text-to-Speech": ["tts_api_endpoint", "tts_model", "tts_voice", "tts_speed",
                                   "stt_min_words", "stt_max_words", "stt_max_char_length", "streamed_playback"]
            }
        )

        if self.is_enabled(False):
            if self.get_plugin_setting("transcribe_audio_enabled"):
                # disable speech-to-text AI model if plugin is enabled
                settings.SetOption("stt_type", "")
                # send available languages for speech-to-text
                settings.SetOption("whisper_languages", self.get_languages())
                websocket.BroadcastMessage(
                    json.dumps({"type": "translate_settings", "data": settings.SETTINGS.get_all_settings()}))
            if self.get_plugin_setting("tts_enabled"):
                # disable speech-to-text AI model if plugin is enabled
                settings.SetOption("tts_enabled", False)

    def get_languages(self):
        languages = {
            "": "Auto",
            **LANGUAGES
        }
        return tuple([{"code": code, "name": language} for code, language in languages.items()])

    def word_char_count_allowed(self, text):
        word_count = len(re.findall(r'\w+', text))
        if self.get_plugin_setting("stt_min_words") == -1 and self.get_plugin_setting(
                "stt_max_words") == -1 and self.get_plugin_setting("stt_max_char_length") == -1:
            return True
        if self.get_plugin_setting("stt_min_words", 1) <= word_count <= self.get_plugin_setting("stt_max_words",
                                                                                                40) and self.get_plugin_setting(
            "stt_max_char_length", 200) >= len(text):
            return True
        else:
            return False

    def _transcribe_audio_api(self, audio, task, language=None):
        url = self.get_plugin_setting("audio_transcribe_api_endpoint")
        audio_model = self.get_plugin_setting("audio_model")

        url = url + "?model=" + audio_model + "&encoding=linear16&sample_rate=16000&smart_format=true&detect_language=true"

        if language is not None and language != "" and language.lower() != "auto":
            url = url + "&language=" + language

        api_key = self.get_plugin_setting("api_key")

        headers = {
            'Authorization': f'Token {api_key}',
            'Content-Type': 'audio/*'
        }

        response = requests.post(url, headers=headers, data=audio)
        if response.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": "Error transcribing text (" + str(
                response.status_code) + "): " + response.text}))
            return "", ""

        response_json = response.json()
        source_language = response_json['results']["channels"][0]["detected_language"]
        transcribed_text = response_json['results']["channels"][0]["alternatives"][0]["transcript"]

        return transcribed_text, source_language

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

    def _tts_api(self, text):
        url = self.get_plugin_setting("tts_api_endpoint")
        api_key = self.get_plugin_setting("api_key")
        tts_voice = self.get_plugin_setting("tts_voice")

        url = url + "?model=" + tts_voice + "&encoding=linear16&sample_rate=" + str(self.tts_source_sample_rate)

        headers = {
            'Authorization': f'Token {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'text': text,
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": "Error generating TTS audio (" + str(
                response.status_code) + "): " + response.text}))
            return ""

        response_data = response.content

        # convert TTS to wav
        raw_data = io.BytesIO()
        save_audio_bytes(response_data, raw_data, "wav")

        # call custom plugin event method
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio',
                                                        {'audio': raw_data, 'sample_rate': self.tts_source_sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            raw_data = plugin_audio['audio']

        return raw_data.getvalue()


    def init_audio_stream_playback(self):
        audio_device = settings.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.GetOption("device_default_out_index")

        if self.audio_streamer is None:
            self.audio_streamer = audio_tools.AudioStreamer(audio_device,
                                                            source_sample_rate=self.tts_source_sample_rate,
                                                            playback_channels=2,
                                                            buffer_size=2048,
                                                            input_channels=1,
                                                            dtype="int16",
                                                            tag="tts",
                                                            )

    def _tts_api_streamed(self, text):
        url = self.get_plugin_setting("tts_api_endpoint")
        api_key = self.get_plugin_setting("api_key")
        tts_voice = self.get_plugin_setting("tts_voice")

        url = url + "?model=" + tts_voice + "&encoding=linear16&sample_rate=" + str(self.tts_source_sample_rate)

        headers = {
            'Authorization': f'Token {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'text': text,
        }

        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
        if response.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": "Error generating TTS audio (" + str(
                response.status_code) + "): " + response.text}))
            return ""

        self.init_audio_stream_playback()
        try:
            # Stream the audio in chunks
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    self.audio_streamer.add_audio_chunk(chunk)
        except Exception as e:
            print(e)

    def on_enable(self):
        self.init()
        pass

    def process_speech_to_text(self, wavefiledata, sample_rate):
        task = settings.GetOption("whisper_task")
        language = settings.GetOption("current_language")
        sample_width = 2  # 2 = 16bit, 4 = 32bit

        # Check if the audio is longer than 0.1 seconds
        audio_duration = len(wavefiledata) / (sample_rate * sample_width)
        if audio_duration <= 0.1:
            print("audio shorter than 0.1 seconds. skipping...")
            return None

        transcribed_text, source_language = self._transcribe_audio_api(wavefiledata, task, language)

        if transcribed_text == "" and source_language == "":
            return None

        result_obj = {
            'text': transcribed_text,
            'type': "transcript",
            'language': source_language
        }
        return result_obj

    def stt_processing(self, audio_data, sample_rate, final_audio) -> dict|None:
        if not self.is_enabled(False) or not self.get_plugin_setting("transcribe_audio_enabled"):
            return None
        if final_audio:
            result_obj = self.process_speech_to_text(audio_data, sample_rate)
            if result_obj is not None:
                return result_obj
        return None

    def stt(self, text, result_obj):
        if self.is_enabled(False) and settings.GetOption("tts_answer") and self.get_plugin_setting(
                "tts_enabled") and text.strip() != "":
            streamed_playback = self.get_plugin_setting("streamed_playback")
            audio_device = settings.GetOption("device_out_index")
            if audio_device is None or audio_device == -1:
                audio_device = settings.GetOption("device_default_out_index")

            if self.word_char_count_allowed(text.strip()):
                if not streamed_playback:
                    wav = self._tts_api(text.strip())
                    if wav is not None:
                        self.play_audio_on_device(wav, audio_device,
                                                  source_sample_rate=self.tts_source_sample_rate,
                                                  audio_device_channel_num=self.tts_target_channels,
                                                  target_channels=self.tts_target_channels,
                                                  input_channels=self.tts_input_channels,
                                                  dtype=self.tts_source_dtype
                                                  )
                else:
                    self._tts_api_streamed(text.strip())
        return

    def tts(self, text, device_index, websocket_connection=None, download=False, path=''):
        if self.is_enabled(False) and self.get_plugin_setting("tts_enabled"):
            streamed_playback = self.get_plugin_setting("streamed_playback")

            if device_index is None or device_index == -1:
                device_index = settings.GetOption("device_default_out_index")

            if not streamed_playback:
                wav = self._tts_api(text.strip())
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
                                                  source_sample_rate=self.tts_source_sample_rate,
                                                  audio_device_channel_num=self.tts_target_channels,
                                                  target_channels=self.tts_target_channels,
                                                  input_channels=self.tts_input_channels,
                                                  dtype=self.tts_source_dtype,
                                                  )
            else:
                if download:
                    wav = self._tts_api(text.strip())
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
                    self._tts_api_streamed(text.strip())
        return


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
            soundfile.write(fp, tempSoundFile.read(), tempSoundFile.samplerate, format=outputFormat)
    else:
        soundfile.write(saveLocation, tempSoundFile.read(), tempSoundFile.samplerate, format=outputFormat)
        if callable(getattr(saveLocation, "flush")):
            saveLocation.flush()
