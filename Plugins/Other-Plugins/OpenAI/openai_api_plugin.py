# ============================================================
# OpenAI API - Whispering Tiger Plugin
# Version 0.0.11
# See https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import base64
import io
import json
import re
import threading
import time
from typing import Union, BinaryIO

import requests
import soundfile

import Plugins
import VRC_OSCLib
import audio_tools
import settings

import websocket

from whisper.tokenizer import LANGUAGES

from Models.TextTranslation import texttranslate

LLM_LANGUAGES = [
    "Albanian",
    "Arabic",
    "Armenian",
    "Awadhi",
    "Azerbaijani",
    "Bashkir",
    "Basque",
    "Belarusian",
    "Bengali",
    "Bhojpuri",
    "Bosnian",
    "Brazilian Portuguese",
    "Bulgarian",
    "Cantonese (Yue)",
    "Catalan",
    "Chhattisgarhi",
    "Chinese",
    "Croatian",
    "Czech",
    "Danish",
    "Dogri",
    "Dutch",
    "English",
    "Estonian",
    "Faroese",
    "Finnish",
    "French",
    "Galician",
    "Georgian",
    "German",
    "Greek",
    "Gujarati",
    "Haryanvi",
    "Hindi",
    "Hungarian",
    "Indonesian",
    "Irish",
    "Italian",
    "Japanese",
    "Javanese",
    "Kannada",
    "Kashmiri",
    "Kazakh",
    "Konkani",
    "Korean",
    "Kyrgyz",
    "Latvian",
    "Lithuanian",
    "Macedonian",
    "Maithili",
    "Malay",
    "Maltese",
    "Mandarin",
    "Mandarin Chinese",
    "Marathi",
    "Marwari",
    "Min Nan",
    "Moldovan",
    "Mongolian",
    "Montenegrin",
    "Nepali",
    "Norwegian",
    "Oriya",
    "Pashto",
    "Persian (Farsi)",
    "Polish",
    "Portuguese",
    "Punjabi",
    "Rajasthani",
    "Romanian",
    "Russian",
    "Sanskrit",
    "Santali",
    "Serbian",
    "Sindhi",
    "Sinhala",
    "Slovak",
    "Slovene",
    "Slovenian",
    "Ukrainian",
    "Urdu",
    "Uzbek",
    "Vietnamese",
    "Welsh",
    "Wu",
]

TTS_VOICES = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer"
]


class OpenAIAPIPlugin(Plugins.Base):
    tts_source_sample_rate = 24000
    tts_source_dtype = "int16"
    tts_input_channels = 1
    tts_target_channels = 2

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "api_key": {"type": "textfield", "value": "", "password": True},

                # text translate settings
                "translate_enabled": False,
                "translate_api_endpoint": "https://api.openai.com/v1/chat/completions",
                "translate_model": {"type": "select", "value": "gpt-3.5",
                                    "values": [
                                        'gpt-3.5-turbo',
                                        'gpt-3.5',
                                        'gpt-4o-mini',
                                        'gpt-4-turbo',
                                        'gpt-4',
                                        'gpt-4o',
                                    ]},
                "translate_temperature": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.7},
                "translate_max_tokens": {"type": "slider", "min": 1, "max": 4096, "step": 1, "value": 64},
                "translate_top_p": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 1},

                # transcribe audio settings
                "transcribe_audio_enabled": False,
                "audio_transcribe_api_endpoint": "https://api.openai.com/v1/audio/transcriptions",
                "audio_translate_api_endpoint": "https://api.openai.com/v1/audio/translations",
                "audio_model": {"type": "select", "value": "whisper-1",
                                "values": [
                                    'whisper-1',
                                ]},

                # text-to-speech settings
                "tts_enabled": False,
                "tts_api_endpoint": "https://api.openai.com/v1/audio/speech",
                "tts_model": {"type": "select", "value": "tts-1",
                              "values": [
                                  'tts-1-hd',
                                  'tts-1',
                              ]},
                "tts_voice": {"type": "select", "value": TTS_VOICES[0],
                              "values": TTS_VOICES},
                "tts_speed": {"type": "slider", "min": 0.25, "max": 4.0, "step": 0.01, "value": 1.0},
                "stt_min_words": -1,
                "stt_max_words": -1,
                "stt_max_char_length": -1,

                # Advanced settings
                "api_key_translate_overwrite": {"type": "textfield", "value": "", "password": True},
                "api_key_speech_to_text_overwrite": {"type": "textfield", "value": "", "password": True},
                "api_key_text_to_speech_overwrite": {"type": "textfield", "value": "", "password": True},
            },
            settings_groups={
                "General": ["api_key", "translate_enabled", "transcribe_audio_enabled", "tts_enabled"],
                "Translate Text": ["translate_api_endpoint", "translate_model",
                                   "translate_temperature", "translate_max_tokens", "translate_top_p"],
                "Speech-to-Text": ["audio_transcribe_api_endpoint",
                                   "audio_translate_api_endpoint", "audio_model"],
                "Text-to-Speech": ["tts_api_endpoint", "tts_model", "tts_voice", "tts_speed",
                                   "stt_min_words", "stt_max_words", "stt_max_char_length"],
                "Advanced": ["api_key_translate_overwrite", "api_key_speech_to_text_overwrite",
                             "api_key_text_to_speech_overwrite"]
            }
        )

        if self.is_enabled(False):
            if self.get_plugin_setting("translate_enabled"):
                # disable txt-translator AI model if plugin is enabled
                settings.SetOption("txt_translator", "")
                # send available languages for text translation
                websocket.BroadcastMessage(
                    json.dumps({"type": "installed_languages", "data": self.return_translation_languages()}))
            if self.get_plugin_setting("transcribe_audio_enabled"):
                # disable speech-to-text AI model if plugin is enabled
                settings.SetOption("stt_type", "")
                # send available languages for speech-to-text
                settings.SetOption("whisper_languages", self.whisper_get_languages())
                websocket.BroadcastMessage(
                    json.dumps({"type": "translate_settings", "data": settings.SETTINGS.get_all_settings()}))
            if self.get_plugin_setting("tts_enabled"):
                # disable speech-to-text AI model if plugin is enabled
                settings.SetOption("tts_enabled", False)

    def whisper_get_languages(self):
        languages = {
            "": "Auto",
            **LANGUAGES
        }
        return tuple([{"code": code, "name": language} for code, language in languages.items()])

    def _translate_text_api(self, text, source_lang, target_lang):
        url = self.get_plugin_setting("translate_api_endpoint")
        api_key = self.get_plugin_setting("api_key")
        if self.get_plugin_setting("api_key_translate_overwrite") != "":
            api_key = self.get_plugin_setting("api_key_translate_overwrite")

        model = self.get_plugin_setting("translate_model")
        temperature = float(self.get_plugin_setting("translate_temperature"))
        max_tokens = int(self.get_plugin_setting("translate_max_tokens"))
        top_p = self.get_plugin_setting("translate_top_p")

        detected_language = source_lang

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': model,
            'messages': [
                {
                    "role": "system",
                    "content": f"You will be provided with a sentence in {source_lang}, and your task is to translate it into {target_lang}."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": "Error translating text (" + str(
                response.status_code) + "): " + response.text}))
            return "", ""

        response_json = response.json()
        translated_text = response_json['choices'][0]["message"]["content"]

        return translated_text, detected_language

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
        if task == "transcribe":
            url = self.get_plugin_setting("audio_transcribe_api_endpoint")
        elif task == "translate":
            url = self.get_plugin_setting("audio_translate_api_endpoint")
        else:
            return "", ""
        audio_model = self.get_plugin_setting("audio_model")

        api_key = self.get_plugin_setting("api_key")
        if self.get_plugin_setting("api_key_speech_to_text_overwrite") != "":
            api_key = self.get_plugin_setting("api_key_speech_to_text_overwrite")

        headers = {
            'Authorization': f'Bearer {api_key}',
            #'Content-Type': 'multipart/form-data'
        }

        files = {
            'file': ('audio.wav', audio, 'audio/wav')
        }
        data = {
            'model': audio_model,
            'response_format': "verbose_json",
        }
        if language is not None and language != "" and language.lower() != "auto" and task == "transcribe":
            data['language'] = language

        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": "Error transcribing text (" + str(
                response.status_code) + "): " + response.text}))
            return "", ""

        response_json = response.json()
        source_language = response_json['language']
        transcribed_text = response_json['text']

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
        tts_model = self.get_plugin_setting("tts_model")
        api_key = self.get_plugin_setting("api_key")
        if self.get_plugin_setting("api_key_text_to_speech_overwrite") != "":
            api_key = self.get_plugin_setting("api_key_text_to_speech_overwrite")
        tts_voice = self.get_plugin_setting("tts_voice")
        tts_speed = self.get_plugin_setting("tts_speed")

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': tts_model,
            'input': text,
            'voice': tts_voice,
            'speed': tts_speed,
            'response_format': "wav",
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

    def text_translate(self, text, from_code, to_code) -> tuple:
        """
        on text_translate event, translates text using OpenAI API.
        """
        if self.is_enabled(False):
            translated_text, detected_language = self._translate_text_api(
                text=text, source_lang=from_code, target_lang=to_code
            )
            return translated_text, detected_language.lower(), to_code
        return "", "", ""

    def return_translation_languages(self):
        return tuple([{"code": language, "name": language} for language in LLM_LANGUAGES])

    def on_plugin_get_languages_call(self, data_obj):
        if self.is_enabled(False):
            data_obj['languages'] = self.return_translation_languages()
            return data_obj

        return None

    def on_enable(self):
        self.init()
        pass

    def _replace_osc_placeholders(self, text, result_obj, settings):
        txt_translate_enabled = settings.GetOption("txt_translate")
        whisper_task = settings.GetOption("whisper_task")

        # replace \n with new line
        text = text.replace("\\n", "\n")

        # replace {src} with source language
        if "language" in result_obj and result_obj["language"] is not None:
            text = text.replace("{src}", result_obj["language"])
        elif "language" in result_obj and result_obj["language"] is None:
            text = text.replace("{src}", "?")

        if txt_translate_enabled and "txt_translation" in result_obj and "txt_translation_target" in result_obj:
            # replace {trg} with target language
            target_language = texttranslate.iso3_to_iso1(result_obj["txt_translation_target"])
            if target_language is None:
                target_language = result_obj["txt_translation_target"]
            if target_language is not None:
                text = text.replace("{trg}", target_language)
        else:
            if "target_lang" in result_obj and result_obj["target_lang"] is not None:
                # replace {trg} with target language of whisper
                text = text.replace("{trg}", result_obj["target_lang"])
            elif whisper_task == "transcribe":
                # replace {trg} with target language of whisper
                if "language" in result_obj and result_obj["language"] is not None:
                    text = text.replace("{trg}", result_obj["language"])
            elif whisper_task == "translate":
                # replace {trg} with target language of whisper
                text = text.replace("{trg}", "en")
            else:
                text = text.replace("{trg}", "?")
        return text

    def _send_message(self, predicted_text, result_obj, final_audio, settings, plugins):
        osc_ip = settings.GetOption("osc_ip")
        osc_address = settings.GetOption("osc_address")
        osc_port = settings.GetOption("osc_port")

        # Update osc_min_time_between_messages option
        VRC_OSCLib.set_min_time_between_messages(settings.GetOption("osc_min_time_between_messages"))

        # WORKAROUND: prevent it from outputting the initial prompt.
        if predicted_text == settings.GetOption("initial_prompt"):
            return

        # Send over OSC
        if osc_ip != "0" and settings.GetOption("osc_auto_processing_enabled") and predicted_text != "":
            osc_notify = final_audio and settings.GetOption("osc_typing_indicator")

            osc_send_type = settings.GetOption("osc_send_type")
            osc_chat_limit = settings.GetOption("osc_chat_limit")
            osc_time_limit = settings.GetOption("osc_time_limit")
            osc_scroll_time_limit = settings.GetOption("osc_scroll_time_limit")
            osc_initial_time_limit = settings.GetOption("osc_initial_time_limit")
            osc_scroll_size = settings.GetOption("osc_scroll_size")
            osc_max_scroll_size = settings.GetOption("osc_max_scroll_size")
            osc_type_transfer_split = settings.GetOption("osc_type_transfer_split")
            osc_type_transfer_split = self._replace_osc_placeholders(osc_type_transfer_split, result_obj, settings)

            osc_text = predicted_text
            if settings.GetOption("osc_type_transfer") == "source":
                osc_text = result_obj["text"]
            elif settings.GetOption("osc_type_transfer") == "both":
                osc_text = result_obj["text"] + osc_type_transfer_split + predicted_text
            elif settings.GetOption("osc_type_transfer") == "both_inverted":
                osc_text = predicted_text + osc_type_transfer_split + result_obj["text"]

            message = self._replace_osc_placeholders(settings.GetOption("osc_chat_prefix"), result_obj, settings) + osc_text

            # delay sending message if it is the final audio and until TTS starts playing
            if final_audio and settings.GetOption("osc_delay_until_audio_playback"):
                # wait until is_audio_playing is True or timeout is reached
                delay_timeout = time.time() + settings.GetOption("osc_delay_timeout")
                tag = settings.GetOption("osc_delay_until_audio_playback_tag")
                tts_answer = settings.GetOption("tts_answer")
                if tag == "tts" and tts_answer:
                    while not audio_tools.is_audio_playing(tag=tag) and time.time() < delay_timeout:
                        time.sleep(0.05)

            if osc_send_type == "full":
                VRC_OSCLib.Chat(message, True, osc_notify, osc_address,
                                IP=osc_ip, PORT=osc_port,
                                convert_ascii=settings.GetOption("osc_convert_ascii"))
            elif osc_send_type == "chunks":
                VRC_OSCLib.Chat_chunks(message,
                                       nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                       chunk_size=osc_chat_limit, delay=osc_time_limit,
                                       initial_delay=osc_initial_time_limit,
                                       convert_ascii=settings.GetOption("osc_convert_ascii"))
            elif osc_send_type == "scroll":
                VRC_OSCLib.Chat_scrolling_chunks(message,
                                                 nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                                 chunk_size=osc_max_scroll_size, delay=osc_scroll_time_limit,
                                                 initial_delay=osc_initial_time_limit,
                                                 scroll_size=osc_scroll_size,
                                                 convert_ascii=settings.GetOption("osc_convert_ascii"))
            elif osc_send_type == "full_or_scroll":
                # send full if message fits in osc_chat_limit, otherwise send scrolling chunks
                if len(message.encode('utf-16le')) <= osc_chat_limit * 2:
                    VRC_OSCLib.Chat(message, True, osc_notify, osc_address,
                                    IP=osc_ip, PORT=osc_port,
                                    convert_ascii=settings.GetOption("osc_convert_ascii"))
                else:
                    VRC_OSCLib.Chat_scrolling_chunks(message,
                                                     nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                                     chunk_size=osc_chat_limit, delay=osc_scroll_time_limit,
                                                     initial_delay=osc_initial_time_limit,
                                                     scroll_size=osc_scroll_size,
                                                     convert_ascii=settings.GetOption("osc_convert_ascii"))

            settings.SetOption("plugin_timer_stopped", True)

    def process_speech_to_text(self, wavefiledata, sample_rate):
        task = settings.GetOption("whisper_task")
        language = settings.GetOption("current_language")

        # convert to wav
        raw_wav_data = audio_tools.audio_bytes_to_wav(wavefiledata, 1, sample_rate)

        transcribed_text, source_language = self._transcribe_audio_api(raw_wav_data, task, language)

        if transcribed_text == "" and source_language == "":
            return

        predicted_text = transcribed_text

        result_obj = {
            'text': transcribed_text,
            'type': "transcript",
            'language': source_language
        }

        # translate using text translator if enabled
        # translate text realtime or after audio is finished
        do_txt_translate = settings.GetOption("txt_translate")
        if do_txt_translate:
            from_lang = settings.GetOption("src_lang")
            to_lang = settings.GetOption("trg_lang")
            to_romaji = settings.GetOption("txt_romaji")
            predicted_text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(transcribed_text, from_lang,
                                                                                         to_lang, to_romaji)
            result_obj["txt_translation"] = predicted_text
            result_obj["txt_translation_source"] = txt_from_lang
            result_obj["txt_translation_target"] = to_lang

        websocket.BroadcastMessage(json.dumps(result_obj))

        self._send_message(predicted_text, result_obj, True, settings.SETTINGS, None)

    def sts(self, wavefiledata, sample_rate):
        if not self.is_enabled(False) or not self.get_plugin_setting("transcribe_audio_enabled"):
            return
        threading.Thread(
            target=self.process_speech_to_text,
            args=(wavefiledata, sample_rate,)
        ).start()
        return

    def stt(self, text, result_obj):
        if self.is_enabled(False) and settings.GetOption("tts_answer") and self.get_plugin_setting(
                "tts_enabled") and text.strip() != "":
            audio_device = settings.GetOption("device_out_index")
            if audio_device is None or audio_device == -1:
                audio_device = settings.GetOption("device_default_out_index")

            if self.word_char_count_allowed(text.strip()):
                wav = self._tts_api(text.strip())
                if wav is not None:
                    self.play_audio_on_device(wav, audio_device,
                                              source_sample_rate=self.tts_source_sample_rate,
                                              audio_device_channel_num=self.tts_target_channels,
                                              target_channels=self.tts_target_channels,
                                              input_channels=self.tts_input_channels,
                                              dtype=self.tts_source_dtype
                                              )
        return

    def tts(self, text, device_index, websocket_connection=None, download=False, path=''):
        if self.is_enabled(False) and self.get_plugin_setting("tts_enabled"):
            if device_index is None or device_index == -1:
                device_index = settings.GetOption("device_default_out_index")

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
