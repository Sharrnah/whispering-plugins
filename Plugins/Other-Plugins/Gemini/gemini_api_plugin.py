# ============================================================
# Gemini API - Whispering Tiger Plugin (REST + Live WS streaming)
# Version 0.0.3
# Docs (for reference):
# - Generate Content:      https://ai.google.dev/api/generate-content
# - Audio Understanding:   https://ai.google.dev/gemini-api/docs/audio
# - Speech (TTS):          https://ai.google.dev/gemini-api/docs/speech-generation
# - Live API (WebSocket):  https://ai.google.dev/gemini-api/docs/live
# ============================================================
import asyncio
import base64
import io
import json
import re
import time
import wave
from typing import Union, BinaryIO, Optional

import requests
import soundfile

import Plugins
import VRC_OSCLib
import audio_tools
import settings

import websocket  # already used in your project
import websockets

from whisper.tokenizer import LANGUAGES
from Models.TextTranslation import texttranslate


LLM_LANGUAGES = [
    "Afrikaans","Albanian","Amharic","Arabic","Armenian","Assamese","Azerbaijani","Basque",
    "Belarusian","Bengali","Bosnian","Bulgarian","Catalan","Cebuano",
    "Chinese (Simplified)","Chinese (Traditional)","Corsican","Croatian","Czech","Danish",
    "Dhivehi","Dutch","English","Esperanto","Estonian","Filipino (Tagalog)","Finnish",
    "French","Frisian","Galician","Georgian","German","Greek","Gujarati","Haitian Creole",
    "Hausa","Hawaiian","Hebrew","Hindi","Hmong","Hungarian","Icelandic","Igbo","Indonesian",
    "Irish","Italian","Japanese","Javanese","Kannada","Kazakh","Khmer","Korean","Kurdish",
    "Kyrgyz","Lao","Latin","Latvian","Lithuanian","Luxembourgish","Macedonian","Malagasy",
    "Malay","Malayalam","Maltese","Maori","Marathi","Mongolian","Myanmar (Burmese)","Nepali",
    "Norwegian","Nyanja (Chichewa)","Odia (Oriya)","Pashto","Persian","Polish","Portuguese",
    "Punjabi","Romanian","Russian","Samoan","Scots Gaelic","Serbian","Sesotho","Shona",
    "Sindhi","Sinhala (Sinhalese)","Slovak","Slovenian","Somali","Spanish","Sundanese",
    "Swahili","Swedish","Tajik","Tamil","Telugu","Thai","Turkish","Ukrainian","Urdu",
    "Uyghur","Uzbek","Vietnamese","Welsh","Xhosa","Yiddish","Yoruba","Zulu",
]

# Gemini voice names (safe subset)
TTS_VOICES = [
    "Kore", "Puck", "Enceladus", "Zephyr", "Aoede", "Leda", "Orus", "Fenrir", "Charon", "Callirrhoe", "Autonoe",
    "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia",
    "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia",
    "Sadaltager", "Sulafat"
]

# Text / multimodal models
LLM_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]

# TTS models (REST one-shot) + LIVE model (WebSocket streaming)
GEMINI_TTS_MODELS = [
    "gemini-2.5-flash-preview-tts",        # REST TTS (full response)
    "gemini-2.5-pro-preview-tts",          # REST TTS (full response)
    #"gemini-live-2.5-flash-preview-native-audio",  # LIVE WS streaming (audio out)
]

# Live WebSocket endpoint for Gemini
LIVE_WS_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"


class GeminiAPIPlugin(Plugins.Base):
    tts_source_sample_rate = 24000  # Gemini TTS output is 24 kHz mono PCM
    tts_source_dtype = "int16"
    tts_input_channels = 1
    tts_target_channels = 2

    audio_streamer = None  # used for streamed playback

    # ---------------------------
    # Init & Settings
    # ---------------------------
    def init(self):
        self.init_plugin_settings(
            {
                "api_key": {"type": "textfield", "value": "", "password": True},
                "refresh_settings_btn": {"label": "Refresh Settings", "type": "button", "style": "primary"},

                # Translate Text
                "translate_enabled": False,
                "translate_api_endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
                "translate_model": {"type": "select", "value": LLM_MODELS[1], "values": LLM_MODELS},
                "translate_temperature": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.7},
                "translate_max_tokens": {"type": "slider", "min": 1, "max": 32768, "step": 1, "value": 256},
                "translate_top_p": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 1.0},

                # Speech-to-Text (Transcribe/Translate)
                "transcribe_audio_enabled": False,
                "audio_transcribe_api_endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
                "audio_translate_api_endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
                "audio_model": {"type": "select", "value": LLM_MODELS[0], "values": LLM_MODELS},

                # Text-to-Speech
                "tts_enabled": False,
                "tts_api_endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
                "tts_model": {"type": "select", "value": GEMINI_TTS_MODELS[0], "values": GEMINI_TTS_MODELS},
                #"tts_voice": {"type": "select", "value": TTS_VOICES[0], "values": TTS_VOICES},
                "tts_speed": {"type": "slider", "min": 0.25, "max": 4.0, "step": 0.01, "value": 1.0},
                "tts_instructions": "",
                "stt_min_words": -1,
                "stt_max_words": -1,
                "stt_max_char_length": -1,

                # Chat
                "chat_api_endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
                "chat_system_prompt": {"type": "textarea", "rows": 5, "value": "You are a helpful assistant."},
                "chat_model": {"type": "select", "value": LLM_MODELS[1], "values": LLM_MODELS},
                "chat_message": {"type": "textarea", "rows": 5, "value": ""},
                "chat_message_send_btn": {"label": "send message", "type": "button", "style": "primary"},
                "chat_temperature": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.7},
                "chat_max_tokens": {"type": "slider", "min": 1, "max": 32768, "step": 1, "value": 256},
                "chat_top_p": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 1.0},

                # Advanced
                "api_key_translate_overwrite": {"type": "textfield", "value": "", "password": True},
                "api_key_speech_to_text_overwrite": {"type": "textfield", "value": "", "password": True},
                "api_key_text_to_speech_overwrite": {"type": "textfield", "value": "", "password": True},
            },
            settings_groups={
                "General": ["api_key", "translate_enabled", "transcribe_audio_enabled", "tts_enabled", "refresh_settings_btn"],
                "Translate Text": ["translate_api_endpoint", "translate_model", "translate_temperature",
                                   "translate_max_tokens", "translate_top_p"],
                "Speech-to-Text": ["audio_transcribe_api_endpoint", "audio_translate_api_endpoint", "audio_model"],
                "Text-to-Speech": ["tts_api_endpoint", "tts_model", "tts_speed", "tts_instructions",
                                   "stt_min_words", "stt_max_words", "stt_max_char_length"],
                "Chat": ["chat_api_endpoint", "chat_model", "chat_message", "chat_system_prompt", "chat_temperature",
                         "chat_max_tokens", "chat_top_p", "chat_message_send_btn"],
                "Advanced": ["api_key_translate_overwrite", "api_key_speech_to_text_overwrite",
                             "api_key_text_to_speech_overwrite"]
            }
        )

        if self.is_enabled(False):
            self.update_settings()

    def update_settings(self):
        if self.get_plugin_setting("translate_enabled"):
            settings.SetOption("txt_translator", "")
            websocket.BroadcastMessage(json.dumps({"type": "installed_languages", "data": self.return_translation_languages()}))
        if self.get_plugin_setting("transcribe_audio_enabled"):
            settings.SetOption("stt_type", "")
            settings.SetOption("whisper_languages", self.whisper_get_languages())
            websocket.BroadcastMessage(json.dumps({"type": "translate_settings", "data": settings.SETTINGS.get_all_settings()}))
        if self.get_plugin_setting("tts_type") != "":
            settings.SetOption("tts_type", "")
            voices_list = []
            for speaker in TTS_VOICES:
                voices_list.append({"name": speaker, "value": speaker})
            websocket.BroadcastMessage(json.dumps({
                "type": "available_tts_voices",
                "data": voices_list
            }))

    def whisper_get_languages(self):
        languages = {"": "Auto", **LANGUAGES}
        return tuple([{"code": code, "name": language} for code, language in languages.items()])

    # ---------------------------
    # Helpers
    # ---------------------------
    def _gemini_headers(self, api_key: str):
        return {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    def _append_model_endpoint(self, base: str, model: str) -> str:
        base = base.rstrip("/")
        return f"{base}/{model}:generateContent"

    def _extract_text(self, resp_json: dict) -> str:
        try:
            return resp_json["candidates"][0]["content"]["parts"][0].get("text", "")
        except Exception:
            return ""

    def _extract_pcm_base64(self, resp_json: dict) -> Optional[str]:
        try:
            return resp_json["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        except Exception:
            return None

    def _pcm_to_wav(self, pcm_bytes: bytes, rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)  # 2 bytes = 16-bit
            wf.setframerate(rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def word_char_count_allowed(self, text):
        word_count = len(re.findall(r'\w+', text))
        if self.get_plugin_setting("stt_min_words") == -1 and self.get_plugin_setting("stt_max_words") == -1 and self.get_plugin_setting("stt_max_char_length") == -1:
            return True
        if self.get_plugin_setting("stt_min_words", 1) <= word_count <= self.get_plugin_setting("stt_max_words", 40) and \
                self.get_plugin_setting("stt_max_char_length", 200) >= len(text):
            return True
        else:
            return False

    # ---------------------------
    # Translate (Text->Text)
    # ---------------------------
    def _translate_text_api(self, text, source_lang, target_lang):
        base = self.get_plugin_setting("translate_api_endpoint")
        api_key = self.get_plugin_setting("api_key")
        if self.get_plugin_setting("api_key_translate_overwrite") != "":
            api_key = self.get_plugin_setting("api_key_translate_overwrite")

        model = self.get_plugin_setting("translate_model")
        temperature = float(self.get_plugin_setting("translate_temperature"))
        max_tokens = int(self.get_plugin_setting("translate_max_tokens"))
        top_p = self.get_plugin_setting("translate_top_p")

        url = self._append_model_endpoint(base, model)
        headers = self._gemini_headers(api_key)

        prompt = (
            f"Translate the following text from {source_lang} to {target_lang}. "
            f"Respond with the translation only, no explanations:\n\n{text}"
        )
        data = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p
            },
            "model": model
        }

        r = requests.post(url, headers=headers, data=json.dumps(data))
        if r.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": f"Error translating text ({r.status_code}): {r.text}"}))
            return "", ""
        resp = r.json()
        translated_text = self._extract_text(resp)
        detected_language = source_lang  # we use the provided hint
        return translated_text, detected_language

    # ---------------------------
    # Speech-to-Text (Transcribe/Translate)
    # ---------------------------
    def _transcribe_audio_api(self, audio_bytes: bytes, task: str, language: Optional[str] = None):
        """
        task: 'transcribe' or 'translate'
        language: optional hint (name, as your UI uses).
        """
        base = self.get_plugin_setting("audio_transcribe_api_endpoint" if task == "transcribe" else "audio_translate_api_endpoint")
        model = self.get_plugin_setting("audio_model")

        api_key = self.get_plugin_setting("api_key")
        if self.get_plugin_setting("api_key_speech_to_text_overwrite") != "":
            api_key = self.get_plugin_setting("api_key_speech_to_text_overwrite")

        url = self._append_model_endpoint(base, model)
        headers = self._gemini_headers(api_key)

        if task == "transcribe":
            instruction = "Transcribe the audio. First line must be ONLY the source language name. Then the full transcript on the following line(s)."
            if language and language.strip() and language.lower() != "auto":
                instruction += f" The language is {language}."
        else:
            instruction = "Translate the audio content into English. Respond with the translation only."

        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": instruction},
                    {
                        "inlineData": {
                            "mimeType": "audio/wav",
                            "data": base64.b64encode(audio_bytes).decode("utf-8")
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 8192
            },
            "model": model
        }

        r = requests.post(url, headers=headers, data=json.dumps(payload))
        if r.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": f"Error transcribing text ({r.status_code}): {r.text}"}))
            return "", ""
        resp = r.json()
        text = self._extract_text(resp)
        if task == "transcribe":
            src_lang = ""
            if text:
                lines = text.splitlines()
                if len(lines) >= 2:
                    src_lang = lines[0].strip()
                    text = "\n".join(lines[1:]).strip()
            return text, (src_lang or "")
        else:
            return text, "en"

    # ---------------------------
    # Audio Playback
    # ---------------------------
    def play_audio_on_device(self, wav, audio_device, source_sample_rate=24000, audio_device_channel_num=2,
                             target_channels=2, input_channels=1, dtype="int16"):
        secondary_audio_device = None
        if settings.GetOption("tts_use_secondary_playback") and (
                (settings.GetOption("tts_secondary_playback_device") == -1 and audio_device != settings.GetOption("device_default_out_index")) or
                (settings.GetOption("tts_secondary_playback_device") > -1 and audio_device != settings.GetOption("tts_secondary_playback_device"))
        ):
            secondary_audio_device = settings.GetOption("tts_secondary_playback_device")
            if secondary_audio_device == -1:
                secondary_audio_device = settings.GetOption("device_default_out_index")

        audio_tools.play_audio(
            wav, audio_device,
            source_sample_rate=source_sample_rate,
            audio_device_channel_num=audio_device_channel_num,
            target_channels=target_channels,
            input_channels=input_channels,
            dtype=dtype,
            secondary_device=secondary_audio_device, tag="tts"
        )

    # ---------------------------
    # TTS (non-streaming) via Gemini TTS preview models (REST)
    # ---------------------------
    def _tts_api(self, text: str) -> bytes:
        base = self.get_plugin_setting("tts_api_endpoint")
        tts_model = self.get_plugin_setting("tts_model")
        api_key = self.get_plugin_setting("api_key")
        if self.get_plugin_setting("api_key_text_to_speech_overwrite") != "":
            api_key = self.get_plugin_setting("api_key_text_to_speech_overwrite")
        tts_voice = settings.GetOption("tts_voice")
        tts_speed = self.get_plugin_setting("tts_speed")
        tts_instructions = self.get_plugin_setting("tts_instructions")

        headers = self._gemini_headers(api_key)
        url = self._append_model_endpoint(base, tts_model)

        # Style/speed hint via natural-language prefix (no numeric rate field here)
        style_prefix = ""
        if tts_instructions:
            style_prefix += tts_instructions.strip() + " "
        if tts_speed and float(tts_speed) != 1.0:
            style_prefix += f"Speak at approximately {float(tts_speed):.2f}x speed. "
        prompt_text = (style_prefix + text).strip()

        data = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": tts_voice
                        }
                    }
                }
            },
            "model": tts_model
        }

        r = requests.post(url, headers=headers, data=json.dumps(data))
        if r.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": f"Error generating TTS audio ({r.status_code}): {r.text}"}))
            return b""

        pcm_b64 = self._extract_pcm_base64(r.json())
        if not pcm_b64:
            return b""

        pcm_bytes = base64.b64decode(pcm_b64)  # 16-bit PCM @ 24kHz mono
        wav_bytes = self._pcm_to_wav(pcm_bytes, rate=self.tts_source_sample_rate, channels=1, sample_width=2)

        # plugin hook
        raw_data = io.BytesIO(wav_bytes)
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': raw_data, 'sample_rate': self.tts_source_sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            raw_data = plugin_audio['audio']
        return raw_data.getvalue()

    # ---------------------------
    # Live API (WebSocket) streaming TTS
    # ---------------------------
    def init_audio_stream_playback(self):
        audio_device = settings.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.GetOption("device_default_out_index")
        if self.audio_streamer is None:
            self.audio_streamer = audio_tools.AudioStreamer(
                audio_device,
                source_sample_rate=self.tts_source_sample_rate,  # 24000
                playback_channels=2,
                buffer_size=2048,
                input_channels=1,
                dtype="int16",
                tag="tts",
            )


    async def _tts_api_streamed_async(self, text: str):
        api_key = self.get_plugin_setting("api_key")
        if self.get_plugin_setting("api_key_text_to_speech_overwrite") != "":
            api_key = self.get_plugin_setting("api_key_text_to_speech_overwrite")

        tts_model = self.get_plugin_setting("tts_model")
        tts_voice = settings.GetOption("tts_voice")
        tts_speed = float(self.get_plugin_setting("tts_speed"))
        tts_instructions = self.get_plugin_setting("tts_instructions")

        if not tts_model.startswith("gemini-live-"):
            # fallback to REST
            wav = self._tts_api(text.strip())
            if wav:
                audio_device = settings.GetOption("device_out_index") or settings.GetOption("device_default_out_index")
                self.play_audio_on_device(wav, audio_device,
                                          source_sample_rate=self.tts_source_sample_rate,
                                          audio_device_channel_num=self.tts_target_channels,
                                          target_channels=self.tts_target_channels,
                                          input_channels=self.tts_input_channels,
                                          dtype=self.tts_source_dtype)
            return

        ws_url = f"{LIVE_WS_URL}?key={api_key}"

        style_prefix = ""
        if tts_instructions:
            style_prefix += tts_instructions.strip() + " "
        if tts_speed and tts_speed != 1.0:
            style_prefix += f"Speak at approximately {tts_speed:.2f}x speed. "
        prompt_text = (style_prefix + text).strip()

        setup_msg = {
            "setup": {
                "model": f"models/{tts_model}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": tts_voice}
                        }
                    }
                }
            }
        }

        self.init_audio_stream_playback()

        async with websockets.connect(ws_url, ping_interval=None, max_size=None) as ws:
            # send setup
            await ws.send(json.dumps(setup_msg))

            # wait for setupComplete
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                if "setupComplete" in msg:
                    break

            # send user text
            client_content = {
                "clientContent": {
                    "turns": [{
                        "role": "user",
                        "parts": [{"text": prompt_text}]
                    }],
                    "turnComplete": True
                }
            }
            await ws.send(json.dumps(client_content))

            # receive audio chunks
            async for raw in ws:
                msg = json.loads(raw)
                if "serverContent" in msg:
                    sc = msg["serverContent"]

                    if "modelTurn" in sc:
                        parts = sc["modelTurn"].get("parts", [])
                        for p in parts:
                            inline = p.get("inlineData")
                            if inline and "data" in inline:
                                pcm_chunk = base64.b64decode(inline["data"])
                                self.audio_streamer.add_audio_chunk(pcm_chunk)

                    if sc.get("generationComplete") or sc.get("turnComplete"):
                        break

    # ---------------------------
    # Public translate hook
    # ---------------------------
    def text_translate(self, text, from_code, to_code) -> tuple:
        if self.is_enabled(False):
            translated_text, detected_language = self._translate_text_api(text=text, source_lang=from_code, target_lang=to_code)
            return translated_text, (detected_language or "").lower(), to_code
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

    # ---------------------------
    # OSC helpers (unchanged semantics)
    # ---------------------------
    def _replace_osc_placeholders(self, text, result_obj, settings_mod):
        txt_translate_enabled = settings_mod.GetOption("txt_translate")
        whisper_task = settings_mod.GetOption("whisper_task")

        text = text.replace("\\n", "\n")

        if "language" in result_obj and result_obj["language"] is not None:
            text = text.replace("{src}", result_obj["language"])
        elif "language" in result_obj and result_obj["language"] is None:
            text = text.replace("{src}", "?")

        if txt_translate_enabled and "txt_translation" in result_obj and "txt_translation_target" in result_obj:
            target_language = texttranslate.iso3_to_iso1(result_obj["txt_translation_target"])
            if target_language is None:
                target_language = result_obj["txt_translation_target"]
            if target_language is not None:
                text = text.replace("{trg}", target_language)
        else:
            if "target_lang" in result_obj and result_obj["target_lang"] is not None:
                text = text.replace("{trg}", result_obj["target_lang"])
            elif whisper_task == "transcribe":
                if "language" in result_obj and result_obj["language"] is not None:
                    text = text.replace("{trg}", result_obj["language"])
            elif whisper_task == "translate":
                text = text.replace("{trg}", "en")
            else:
                text = text.replace("{trg}", "?")
        return text

    def _send_message(self, predicted_text, result_obj, final_audio, settings_mod, plugins):
        osc_ip = settings_mod.GetOption("osc_ip")
        osc_address = settings_mod.GetOption("osc_address")
        osc_port = settings_mod.GetOption("osc_port")

        VRC_OSCLib.set_min_time_between_messages(settings_mod.GetOption("osc_min_time_between_messages"))

        if predicted_text == settings_mod.GetOption("initial_prompt"):
            return

        if osc_ip != "0" and settings_mod.GetOption("osc_auto_processing_enabled") and predicted_text != "":
            osc_notify = final_audio and settings_mod.GetOption("osc_typing_indicator")

            osc_send_type = settings_mod.GetOption("osc_send_type")
            osc_chat_limit = settings_mod.GetOption("osc_chat_limit")
            osc_time_limit = settings_mod.GetOption("osc_time_limit")
            osc_scroll_time_limit = settings_mod.GetOption("osc_scroll_time_limit")
            osc_initial_time_limit = settings_mod.GetOption("osc_initial_time_limit")
            osc_scroll_size = settings_mod.GetOption("osc_scroll_size")
            osc_max_scroll_size = settings_mod.GetOption("osc_max_scroll_size")
            osc_type_transfer_split = settings_mod.GetOption("osc_type_transfer_split")
            osc_type_transfer_split = self._replace_osc_placeholders(osc_type_transfer_split, result_obj, settings_mod)

            osc_text = predicted_text
            if settings_mod.GetOption("osc_type_transfer") == "source":
                osc_text = result_obj["text"]
            elif settings_mod.GetOption("osc_type_transfer") == "both":
                osc_text = result_obj["text"] + osc_type_transfer_split + predicted_text
            elif settings_mod.GetOption("osc_type_transfer") == "both_inverted":
                osc_text = predicted_text + osc_type_transfer_split + result_obj["text"]

            message = self._replace_osc_placeholders(settings_mod.GetOption("osc_chat_prefix"), result_obj, settings_mod) + osc_text

            if final_audio and settings_mod.GetOption("osc_delay_until_audio_playback"):
                delay_timeout = time.time() + settings_mod.GetOption("osc_delay_timeout")
                tag = settings_mod.GetOption("osc_delay_until_audio_playback_tag")
                tts_answer = settings_mod.GetOption("tts_answer")
                if tag == "tts" and tts_answer:
                    while not audio_tools.is_audio_playing(tag=tag) and time.time() < delay_timeout:
                        time.sleep(0.05)

            if osc_send_type == "full":
                VRC_OSCLib.Chat(message, True, osc_notify, osc_address, IP=osc_ip, PORT=osc_port,
                                convert_ascii=settings_mod.GetOption("osc_convert_ascii"))
            elif osc_send_type == "chunks":
                VRC_OSCLib.Chat_chunks(message, nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                       chunk_size=osc_chat_limit, delay=osc_time_limit,
                                       initial_delay=osc_initial_time_limit,
                                       convert_ascii=settings_mod.GetOption("osc_convert_ascii"))
            elif osc_send_type == "scroll":
                VRC_OSCLib.Chat_scrolling_chunks(message, nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                                 chunk_size=osc_max_scroll_size, delay=osc_scroll_time_limit,
                                                 initial_delay=osc_initial_time_limit,
                                                 scroll_size=osc_scroll_size,
                                                 convert_ascii=settings_mod.GetOption("osc_convert_ascii"))
            elif osc_send_type == "full_or_scroll":
                if len(message.encode('utf-16le')) <= osc_chat_limit * 2:
                    VRC_OSCLib.Chat(message, True, osc_notify, osc_address, IP=osc_ip, PORT=osc_port,
                                    convert_ascii=settings_mod.GetOption("osc_convert_ascii"))
                else:
                    VRC_OSCLib.Chat_scrolling_chunks(message, nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                                     chunk_size=osc_chat_limit, delay=osc_scroll_time_limit,
                                                     initial_delay=osc_initial_time_limit,
                                                     scroll_size=osc_scroll_size,
                                                     convert_ascii=settings_mod.GetOption("osc_convert_ascii"))
            settings_mod.SetOption("plugin_timer_stopped", True)

    # ---------------------------
    # STT pipeline wrappers
    # ---------------------------
    def process_speech_to_text(self, wavefiledata, sample_rate):
        task = settings.GetOption("whisper_task")  # 'transcribe' or 'translate'
        language = settings.GetOption("current_language")
        sample_width = 2

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

    # ---------------------------
    # TTS trigger (answer speech)
    # ---------------------------
    def stt(self, text, result_obj):
        if self.is_enabled(False) and settings.GetOption("tts_answer") and self.get_plugin_setting("tts_enabled") and text.strip() != "":
            audio_device = settings.GetOption("device_out_index")
            if audio_device is None or audio_device == -1:
                audio_device = settings.GetOption("device_default_out_index")

            if self.word_char_count_allowed(text.strip()):
                if settings.GetOption("tts_streamed_playback"):
                    asyncio.run(self._tts_api_streamed_async(text.strip()))
                else:
                    wav = self._tts_api(text.strip())
                    if wav:
                        self.play_audio_on_device(
                            wav, audio_device,
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

            if settings.GetOption("tts_streamed_playback") and not download:
                # true streaming path; playback happens inside the streamer loop
                asyncio.run(self._tts_api_streamed_async(text.strip()))
            else:
                # non-streamed (REST) path / save-to-file path
                wav = self._tts_api(text.strip())
                if wav:
                    if download:
                        if path:
                            with open(path, "wb") as f:
                                f.write(wav)
                            websocket.BroadcastMessage(json.dumps({"type": "info", "data": "File saved to: " + path}))
                        else:
                            if websocket_connection is not None:
                                wav_data = base64.b64encode(wav).decode('utf-8')
                                websocket.AnswerMessage(websocket_connection, json.dumps({"type": "tts_save", "wav_data": wav_data}))
                    else:
                        self.play_audio_on_device(
                            wav, device_index,
                            source_sample_rate=self.tts_source_sample_rate,
                            audio_device_channel_num=self.tts_target_channels,
                            target_channels=self.tts_target_channels,
                            input_channels=self.tts_input_channels,
                            dtype=self.tts_source_dtype
                        )
        return

    # ---------------------------
    # Chat message (LLM)
    # ---------------------------
    def chat_message_process(self):
        api_key = self.get_plugin_setting("api_key")
        base = self.get_plugin_setting("chat_api_endpoint")
        chat_system_prompt = self.get_plugin_setting("chat_system_prompt")
        chat_message = self.get_plugin_setting("chat_message")
        if chat_message == "":
            return

        model = self.get_plugin_setting("chat_model")
        temperature = float(self.get_plugin_setting("chat_temperature"))
        max_tokens = int(self.get_plugin_setting("chat_max_tokens"))
        top_p = self.get_plugin_setting("chat_top_p")

        url = self._append_model_endpoint(base, model)
        headers = self._gemini_headers(api_key)

        data = {
            "model": model,
            "systemInstruction": {"role": "system", "parts": [{"text": chat_system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": chat_message}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p
            }
        }

        r = requests.post(url, headers=headers, data=json.dumps(data))
        if r.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": f"Error getting chat completion ({r.status_code}): {r.text}"}))
            return

        response_json = r.json()
        response_text = self._extract_text(response_json)
        if response_text.strip() == "":
            return
        settings.SetOption("websocket_final_messages", False)
        result_obj = {'text': chat_message, 'llm_answer': response_text, 'type': "llm_answer"}
        websocket.BroadcastMessage(json.dumps(result_obj))
        settings.SetOption("websocket_final_messages", True)

    # ---------------------------
    # UI event
    # ---------------------------
    def on_event_received(self, message, websocket_connection=None):
        if "type" not in message:
            return
        if message["type"] == "plugin_button_press":
            if message["value"] == "chat_message_send_btn":
                self.chat_message_process()

            if message["value"] == "refresh_settings_btn":
                self.update_settings()
        pass


# ---------------------------
# Utility kept from your file
# ---------------------------
def save_audio_bytes(audioData: bytes, saveLocation: Union[BinaryIO, str], outputFormat) -> None:
    """
    Saves audio bytes (with an audio container) to a location or file-like object.
    NOTE: For Gemini native TTS we convert PCM->WAV separately before calling this.
    """
    tempSoundFile = soundfile.SoundFile(io.BytesIO(audioData))
    if isinstance(saveLocation, str):
        with open(saveLocation, "wb") as fp:
            soundfile.write(fp, tempSoundFile.read(), tempSoundFile.samplerate, format=outputFormat)
    else:
        soundfile.write(saveLocation, tempSoundFile.read(), tempSoundFile.samplerate, format=outputFormat)
        if callable(getattr(saveLocation, "flush", None)):
            saveLocation.flush()
