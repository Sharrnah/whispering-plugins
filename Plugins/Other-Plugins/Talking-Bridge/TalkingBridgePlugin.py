# ============================================================
# Talking Bridge Plugin for Whispering Tiger
# V0.0.9
# See https://github.com/Sharrnah/whispering-ui
# Translates dynamically speech between languages
# ============================================================
#
import io
import threading
import time
import traceback
import wave

import soundfile
from pydub import AudioSegment

import Plugins
import VRC_OSCLib
import audio_tools
import settings
from Models.TTS import tts

from Models.TextTranslation import texttranslate
from Models import languageClassification

from Utilities.iso_converter import LanguageCodeConverter

class TalkingBridgePlugin(Plugins.Base):
    audio_model = None
    #last_recorded_chunk_time = None

    last_audio = None

    audio_file_target_sample_rate = 44000
    LAST_STT_TIME = time.time()
    advert_thread = None

    language_code_converter = LanguageCodeConverter()

    def init(self):
        # get STT languages
        whisper_languages = settings.SETTINGS.GetOption("whisper_languages")
        whisper_languages_list = [["", ""]]
        if whisper_languages is not None and len(whisper_languages) > 0:
            whisper_languages_list = [[lang['name'], lang['code']] for lang in whisper_languages]

        # get text translation languages
        source_text_translation_languages = []
        target_text_translation_languages = []
        texttranslate_languages = texttranslate.GetInstalledLanguageNames()
        if texttranslate_languages is not None:
            source_text_translation_languages = [[lang['name'], lang['code']] for lang in texttranslate_languages]
            #source_text_translation_languages.insert(0, ["Auto", "auto"])
            target_text_translation_languages = [[lang['name'], lang['code']] for lang in texttranslate_languages]

        voices_list = [""]
        if self.is_enabled(False):
            settings.SETTINGS.SetOption("osc_auto_processing_enabled", False)
            settings.SETTINGS.SetOption("osc_force_activity_indication", True)

            if tts.init():
                voices_list = tts.tts.list_voices()
                if settings.SETTINGS.GetOption("tts_type") == "zonos":
                    voices_list.insert(0, "auto_clone")

        self.init_plugin_settings(
            {
                # General
                "osc_enabled": False,
                "tts_enabled": False,
                "translation_enabled": False,
                "first_speaker_language": {"type": "select_completion", "value": "", "values": whisper_languages_list},
                "first_source_language": {"type": "select_completion", "value": "", "values": source_text_translation_languages},
                "first_target_language": {"type": "select_completion", "value": "", "values": target_text_translation_languages},
                "first_target_voice": {"type": "select", "value": "", "values": voices_list},
                "second_speaker_language": {"type": "select_completion", "value": "", "values": whisper_languages_list},
                "second_source_language": {"type": "select_completion", "value": "", "values": source_text_translation_languages},
                "second_target_language": {"type": "select_completion", "value": "", "values": target_text_translation_languages},
                "second_target_voice": {"type": "select", "value": "", "values": voices_list},
                "advert_active": False,
                "advert_inactivity_time": {"type": "slider", "min": 0, "max": 300, "step": 1, "value": 100},
                "advert_frequency": {"type": "slider", "min": 0, "max": 300, "step": 1, "value": 50},
                "advert_text": {"type": "textarea", "rows": 6, "value": "I auto translate between Japanese and English.\n\nStay close and speak.\nWait for translation to finish before speaking again.\n\n🐅 Whispering Tiger\nhttps://whispering-tiger.github.io/"},
                "advert_text_2": {"type": "textarea", "rows": 6, "value": "私は日本語と英語のあいだを自動翻訳します。\n\n近くにいて、話しかけて。\n翻訳が終わるまで話さずにお待ちください。\n\n🐅 Whispering Tiger\nhttps://whispering-tiger.github.io/"},
                "advert_audio": {"type": "file_open", "accept": ".wav,.mp3", "value": ""},
                "fallback_action": {"type": "select_textvalue", "value": "", "values":
                    [
                        ["Do nothing", ""],
                        ["Translate to First language", "ToFirst"],
                        ["Translate to Second language", "ToSecond"],
                        ["Play audio and OSC Text", "playAudio"]
                    ]
                },
                "fallback_audio": {"type": "file_open", "accept": ".wav,.mp3", "value": ""},
                "fallback_text": {"type": "textarea", "rows": 3, "value": "Sorry, i only speak English and Japanese"},
                "advert_info": {"type": "label", "label": "", "style": "left"},
                "language_probability_threshold": {"type": "slider", "min": 0, "max": 1, "step": 0.1, "value": 0.2},
            },
            settings_groups={
                "General": ["osc_enabled", "tts_enabled", "translation_enabled"],
                "Languages": [
                    ["first_speaker_language", "second_speaker_language", "first_target_voice"],
                    ["first_source_language", "second_source_language", "second_target_voice"],
                    ["first_target_language", "second_target_language", "language_probability_threshold"],
                ],
                "Fallback": [
                    "fallback_action", "fallback_audio", "fallback_text"
                ],
                "Advertisement": [
                    ["advert_active",   "advert_frequency",       "advert_text", "advert_text_2"],
                    ["advert_info", "advert_inactivity_time", "advert_audio"],
                ],
            }
        )

        self.on_enable()

        pass


    def convert_mp3_to_wav(self, mp3_path):
        audio = AudioSegment.from_mp3(mp3_path)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io

    def pcm_to_wav_bytes(self, pcm_bytes: bytes, sample_rate: int, channels: int, sampwidth: int):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)   # e.g. 2 for 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        buf.seek(0)
        return buf

    def play_audio_on_device(self, wav, audio_device, source_sample_rate=22050, audio_device_channel_num=2,
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

        stop_play = not self.get_plugin_setting("allow_overlapping_audio")

        audio_tools.play_audio(wav, audio_device,
                               source_sample_rate=source_sample_rate,
                               audio_device_channel_num=audio_device_channel_num,
                               target_channels=target_channels,
                               input_channels=input_channels,
                               dtype=dtype,
                               secondary_device=secondary_audio_device,
                               stop_play=stop_play, tag="tts")

    def play_audio_file(self, file_path=""):
        if file_path != "":
            if file_path.endswith(".wav"):
                wav_numpy = audio_tools.load_wav_to_bytes(file_path, target_sample_rate=self.audio_file_target_sample_rate)
            elif file_path.endswith(".mp3"):
                mp3_file_obj = self.convert_mp3_to_wav(file_path)
                wav_numpy = audio_tools.load_wav_to_bytes(mp3_file_obj, target_sample_rate=self.audio_file_target_sample_rate)
            else:
                # unsupported file format
                return

            # Convert numpy array back to WAV bytes
            with io.BytesIO() as byte_io:
                soundfile.write(byte_io, wav_numpy, samplerate=self.audio_file_target_sample_rate,
                                format='WAV')  # Explicitly specify format
                wav_bytes = byte_io.getvalue()

            self.play_audio_on_device(wav_bytes, settings.GetOption("device_out_index"),
                                      source_sample_rate=self.audio_file_target_sample_rate,
                                      audio_device_channel_num=2,
                                      target_channels=2,
                                      input_channels=1,
                                      dtype="int16")

    def sts(self, wavefiledata, sample_rate):
        if self.is_enabled():
            self.LAST_STT_TIME = time.time()
            wav_buf = self.pcm_to_wav_bytes(
                wavefiledata,   # your raw PCM bytes
                sample_rate=16000,
                channels=1,
                sampwidth=2     # 16-bit PCM = 2 bytes
            )
            self.last_audio = wav_buf

    def stt_intermediate(self, text, result_obj):
        if self.is_enabled():
            self.LAST_STT_TIME = time.time()

    def advert_thread_func(self):
        last_advert_time = 0
        last_advert_text = 0
        while self.is_enabled():
            advert_inactivity_time = self.get_plugin_setting("advert_inactivity_time")
            advert_frequency = self.get_plugin_setting("advert_frequency")
            advert_text_1 = self.get_plugin_setting("advert_text")
            advert_text_2 = self.get_plugin_setting("advert_text_2")
            advert_audio = self.get_plugin_setting("advert_audio")

            if advert_frequency == 0 or not self.get_plugin_setting("advert_active"):
                time.sleep(100)
                continue

            if advert_inactivity_time < (time.time() - self.LAST_STT_TIME):
                if (time.time() - last_advert_time) > advert_frequency:
                    last_advert_time = time.time()

                    if last_advert_text == 1 and advert_text_2 != "":
                        advert_text = advert_text_2
                        last_advert_text = 2
                    else:
                        advert_text = advert_text_1
                        last_advert_text = 1

                    if advert_text != "" and self.get_plugin_setting("osc_enabled"):
                        osc_ip = settings.SETTINGS.GetOption("osc_ip")
                        osc_port = settings.SETTINGS.GetOption("osc_port")
                        osc_address = settings.SETTINGS.GetOption("osc_address")
                        osc_chat_limit = settings.SETTINGS.GetOption("osc_chat_limit")
                        osc_time_limit = settings.SETTINGS.GetOption("osc_time_limit")
                        osc_initial_time_limit = settings.SETTINGS.GetOption("osc_initial_time_limit")
                        osc_convert_ascii = settings.SETTINGS.GetOption("osc_convert_ascii")

                        VRC_OSCLib.Chat_chunks(advert_text,
                                               nofify=False, address=osc_address, ip=osc_ip, port=osc_port,
                                               chunk_size=osc_chat_limit, delay=osc_time_limit,
                                               initial_delay=osc_initial_time_limit,
                                               convert_ascii=osc_convert_ascii)
                    if advert_audio != "":
                        self.play_audio_file(advert_audio)
            time.sleep(1)
        pass

    def on_enable(self):
        if self.is_enabled(False):
            self.advert_thread = threading.Thread(target=self.advert_thread_func, daemon=True)
            self.advert_thread.start()

    def on_disable(self):
        if not self.is_enabled(False):
            if self.advert_thread is not None and self.advert_thread.is_alive():
                self.advert_thread.join(timeout=1)
            self.advert_thread = None

    def get_all_special_settings(self, type):
        current = settings.GetOption("special_settings") or {}
        special_settings = dict(current)
        if type in special_settings:
            return special_settings[type]
        return {}

    def get_special_settings(self, type, name):
        current = settings.GetOption("special_settings") or {}
        special_settings = dict(current)
        if type in special_settings and name in special_settings[type]:
            return special_settings[type][name]
        return None

    def update_special_settings(self, type, name, value):
        current = settings.GetOption("special_settings") or {}
        special_settings = dict(current)
        if type not in special_settings:
            special_settings[type] = {}
        special_settings[type][name] = value
        settings.SetOption("special_settings", special_settings)

    def determine_tts_settings(self, language):
        if settings.SETTINGS.GetOption("tts_type") == "kokoro":
            #special_settings = tts.tts.special_settings
            special_settings = self.get_all_special_settings("tts_kokoro")
            #self.update_special_settings("tts_kokoro", "language", language)
            tts_language = 'a'
            match language.lower():
                case 'e' | 'en' | 'eng' | 'english' | 'en-us' | 'en_us' | 'eng_latn':
                    tts_language = 'a'
                case 'es' | 'esp' | 'spanish' | 'es-es' | 'es_es' | 'spa_latn':
                    tts_language = 'e'
                case 'f' | 'fr' | 'fra' | 'french' | 'fr-fr' | 'fr_fr' | 'fra_latn':
                    tts_language = 'f'
                case 'h' | 'hi' | 'hin' | 'hindi' | 'hi-in' | 'hi_in' | 'hin_deva':
                    tts_language = 'h'
                case 'i' | 'it' | 'ita' | 'italian' | 'it-it' | 'it_it' | 'ita_latn':
                    tts_language = 'i'
                case 'j' | 'ja' | 'jp' | 'jpn' | 'japanese' | 'ja-jp' | 'ja_jp' | 'jpn_jpan':
                    tts_language = 'j'
                case 'b' | 'br' | 'bra' | 'brazilian_portuguese' | 'portuguese' | 'pt' | 'pt-br' | 'pt_br' | 'por_latn':
                    tts_language = 'p'
                case 'z' | 'zh' | 'zho' | 'cn' | 'chinese' | 'zh-cn' | 'zh_cn' | 'mandarin' | 'zho_hans' | 'zho_hant':
                    tts_language = 'z'
            special_settings["language"] = tts_language
            #tts.tts.set_special_setting(special_settings)
            self.update_special_settings("tts_kokoro", "language", tts_language)
        if settings.SETTINGS.GetOption("tts_type") == "zonos":
            #special_settings = tts.tts.special_settings
            special_settings = self.get_all_special_settings("tts_zonos")
            tts_language = 'en-us'
            match language.lower():
                case 'e' | 'en' | 'eng' | 'english' | 'en-us' | 'en_us' | 'eng_latn':
                    tts_language = 'en-us'
                case 'd' | 'de' | 'deu' | 'german' | 'de-de' | 'de_de' | 'deu_latn':
                    tts_language = 'de-de'
                case 'es' | 'esp' | 'spanish' | 'es-es' | 'es_es' | 'spa_latn':
                    tts_language = 'es'
                case 'f' | 'fr' | 'fra' | 'french' | 'fr-fr' | 'fr_fr' | 'fra_latn':
                    tts_language = 'fr-fr'
                case 'h' | 'hi' | 'hin' | 'hindi' | 'hi-in' | 'hi_in' | 'hin_deva':
                    tts_language = 'hi'
                case 'i' | 'it' | 'ita' | 'italian' | 'it-it' | 'it_it' | 'ita_latn':
                    tts_language = 'it'
                case 'j' | 'ja' | 'jp' | 'jpn' | 'japanese' | 'ja-jp' | 'ja_jp' | 'jpn_jpan':
                    tts_language = 'ja'
                case 'b' | 'br' | 'bra' | 'brazilian_portuguese' | 'portuguese' | 'pt' | 'pt-br' | 'pt_br' | 'por_latn':
                    tts_language = 'pt-br'
                case 'z' | 'zh' | 'zho' | 'cn' | 'chinese' | 'zh-cn' | 'zh_cn' | 'mandarin' | 'zho_hans' | 'zho_hant':
                    tts_language = 'cmn'
            special_settings["language"] = tts_language
            #tts.tts.set_special_setting(special_settings)
            self.update_special_settings("tts_zonos", "language", tts_language)

    def run_tts(self, text, voice=""):
        audio_device = settings.SETTINGS.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.SETTINGS.GetOption("device_default_out_index")

        voice_ref = None
        if voice != "" and voice != "auto_clone":
            settings.SETTINGS.SetOption("tts_voice", voice)
            voice_ref = None

        if voice == "auto_clone":
            voice_ref = self.last_audio

        if tts.init():
            streamed_playback = settings.SETTINGS.GetOption("tts_streamed_playback")
            tts_wav = None

            if hasattr(tts.tts, "enqueue_tts"):
                tts.tts.enqueue_tts(text, streamed_playback, voice_ref)
            else:
                if streamed_playback and hasattr(tts.tts, "tts_streaming"):
                    #if hasattr(tts.tts, "stop"):
                    #    tts.tts.stop()
                    tts_wav, sample_rate = tts.tts.tts_streaming(text, voice_ref)

                if tts_wav is None:
                    streamed_playback = False
                    tts_wav, sample_rate = tts.tts.tts(text, voice_ref)

                if tts_wav is not None and not streamed_playback:
                    tts.tts.play_audio(tts_wav, audio_device)
        else:
            for plugin_inst in Plugins.plugins:
                if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'tts'):
                    try:
                        plugin_inst.tts(text, audio_device, None, False, '')
                    except Exception as e:
                        print(f"Plugin TTS failed in Plugin {plugin_inst.__class__.__name__}:", e)
                        traceback.print_exc()
        pass

    def stt(self, text, result_obj):
        if self.is_enabled():
            osc_ip = settings.SETTINGS.GetOption("osc_ip")
            osc_port = settings.SETTINGS.GetOption("osc_port")
            osc_address = settings.SETTINGS.GetOption("osc_address")
            osc_chat_limit = settings.SETTINGS.GetOption("osc_chat_limit")
            osc_time_limit = settings.SETTINGS.GetOption("osc_time_limit")
            osc_initial_time_limit = settings.SETTINGS.GetOption("osc_initial_time_limit")
            osc_convert_ascii = settings.SETTINGS.GetOption("osc_convert_ascii")

            self.LAST_STT_TIME = time.time()
            speaker_lang = result_obj["language"]
            language_probability_threshold = self.get_plugin_setting("language_probability_threshold")

            # @todo lid detected language has to fit with text language also make probability configurable
            language_code_type = self.language_code_converter.any_to_code_type(self.get_plugin_setting("first_source_language"))[0]
            text_lang = ""

            if "language_probability" in result_obj and result_obj["language_probability"] <= language_probability_threshold:
                speaker_lang_probability = result_obj["language_probability"]
                try:
                    text_lang_detected, text_lang_probability = languageClassification.classify(text, to_code=language_code_type)
                    if text_lang_probability > speaker_lang_probability:
                        text_lang = text_lang_detected
                        print("Text language detected:", text_lang)
                except Exception as e:
                    print("Text language detection failed:", e)
            else:
                try:
                    text_lang_detected, text_lang_probability = languageClassification.classify(text, to_code=language_code_type)
                    if text_lang_probability >= language_probability_threshold:
                        text_lang = text_lang_detected
                        print("Text language detected B:", text_lang)
                except Exception as e:
                    print("Text language detection failed B:", e)

            # Determine target language
            target_lang = "en"
            source_lang = "en"
            target_voice = ""
            if (text_lang != "" and text_lang == self.get_plugin_setting("first_source_language")) or speaker_lang == self.get_plugin_setting("first_speaker_language"):
                source_lang = self.get_plugin_setting("first_source_language")
                target_lang = self.get_plugin_setting("first_target_language")
                target_voice = self.get_plugin_setting("first_target_voice") or settings.GetOption("tts_voice")
            elif (text_lang != "" and text_lang == self.get_plugin_setting("second_source_language")) or speaker_lang == self.get_plugin_setting("second_speaker_language"):
                source_lang = self.get_plugin_setting("second_source_language")
                target_lang = self.get_plugin_setting("second_target_language")
                target_voice = self.get_plugin_setting("second_target_voice") or settings.GetOption("tts_voice")
            else:
                fallback_action = self.get_plugin_setting("fallback_action")
                match fallback_action:
                    case "ToFirst":
                        source_lang = "auto"
                        target_lang = self.get_plugin_setting("first_target_language")
                        target_voice = self.get_plugin_setting("first_target_voice") or settings.GetOption("tts_voice")
                    case "ToSecond":
                        source_lang = "auto"
                        target_lang = self.get_plugin_setting("second_target_language")
                        target_voice = self.get_plugin_setting("second_target_voice") or settings.GetOption("tts_voice")
                    case "playAudio":
                        fallback_audio = self.get_plugin_setting("fallback_audio")
                        fallback_text = self.get_plugin_setting("fallback_text")
                        if fallback_audio != "" and self.get_plugin_setting("tts_enabled"):
                            self.play_audio_file(fallback_audio)
                        if fallback_text != "" and self.get_plugin_setting("osc_enabled"):
                            VRC_OSCLib.Chat_chunks(fallback_text,
                                                   nofify=True, address=osc_address, ip=osc_ip, port=osc_port,
                                                   chunk_size=osc_chat_limit, delay=osc_time_limit,
                                                   initial_delay=osc_initial_time_limit,
                                                   convert_ascii=osc_convert_ascii)
                        return
                    case _:
                        return


            to_code = speaker_lang
            translation_text = text
            if self.get_plugin_setting("translation_enabled"):
                translation_text, from_code, to_code = texttranslate.TranslateLanguage(text, source_lang, target_lang)
                print("to_code:", to_code)

            if self.get_plugin_setting("osc_enabled"):

                VRC_OSCLib.Chat_chunks(translation_text,
                                       nofify=True, address=osc_address, ip=osc_ip, port=osc_port,
                                       chunk_size=osc_chat_limit, delay=osc_time_limit,
                                       initial_delay=osc_initial_time_limit,
                                       convert_ascii=osc_convert_ascii)

            if self.get_plugin_setting("tts_enabled"):
                self.determine_tts_settings(to_code)
                if target_voice != "":
                    settings.SETTINGS.SetOption("tts_voice", target_voice)
                self.run_tts(translation_text, voice=target_voice)
        return
