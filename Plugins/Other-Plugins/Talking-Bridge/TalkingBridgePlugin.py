# ============================================================
# Talking Bridge Plugin for Whispering Tiger
# V0.0.2
# See https://github.com/Sharrnah/whispering-ui
# Translates dynamically speech between languages
# ============================================================
#
import time
import traceback

import Plugins
import VRC_OSCLib
import settings
from Models.TTS import tts
from whisper.tokenizer import LANGUAGES

from Models.TextTranslation import texttranslate

class TalkingBridgePlugin(Plugins.Base):
    audio_model = None
    last_recorded_chunk_time = time.time()

    def init(self):
        whisper_languages = sorted(LANGUAGES.keys())
        whisper_languages.insert(0, "auto")

        text_translation_languages = []
        texttranslate_languages = texttranslate.GetInstalledLanguageNames()
        if texttranslate_languages is not None:
            text_translation_languages = [lang['code'] for lang in texttranslate_languages]
        source_text_translation_languages = list(text_translation_languages)
        text_translation_languages.insert(0, "")
        source_text_translation_languages.insert(0, "auto")

        self.init_plugin_settings(
            {
                # General
                "osc_enabled": False,
                "tts_enabled": False,
                "translation_enabled": False,
                "first_speaker_language": {"type": "select", "value": "", "values": whisper_languages},
                "first_source_language": {"type": "select", "value": "", "values": source_text_translation_languages},
                "first_target_language": {"type": "select", "value": "", "values": text_translation_languages},
                "second_speaker_language": {"type": "select", "value": "", "values": whisper_languages},
                "second_source_language": {"type": "select", "value": "", "values": source_text_translation_languages},
                "second_target_language": {"type": "select", "value": "", "values": text_translation_languages},
            },
            settings_groups={
                "General": ["osc_enabled", "tts_enabled", "translation_enabled"],
                "Languages": [
                    ["first_speaker_language", "second_speaker_language"],
                    ["first_source_language", "second_source_language"],
                    ["first_target_language", "second_target_language"],
                ],
            }
        )
        if self.is_enabled(False):
            settings.SETTINGS.SetOption("osc_auto_processing_enabled", False)
        pass

    def stt_intermediate(self, text, result_obj):
        current_spoken_time = time.time()
        if self.get_plugin_setting("osc_enabled"):
            osc_ip = settings.SETTINGS.GetOption("osc_ip")
            osc_port = settings.SETTINGS.GetOption("osc_port")

            if current_spoken_time - self.last_recorded_chunk_time > 2.0:
                self.last_recorded_chunk_time = current_spoken_time

            VRC_OSCLib.Bool(True, "/chatbox/typing", IP=osc_ip, PORT=osc_port)

    def determine_tts_settings(self, language):
        if settings.SETTINGS.GetOption("tts_type") == "kokoro":
            tts_language = 'a'
            tts_voice = 'af_heart'
            match language.lower():
                case 'e' | 'en' | 'eng' | 'english' | 'en-us' | 'en_us' | 'eng_latn':
                    tts_language = 'a'
                    tts_voice = 'af_bella'
                case 'es' | 'esp' | 'spanish' | 'es-es' | 'es_es' | 'spa_latn':
                    tts_language = 'e'
                    tts_voice = 'ef_dora'
                case 'f' | 'fr' | 'fra' | 'french' | 'fr-fr' | 'fr_fr' | 'fra_latn':
                    tts_language = 'f'
                    tts_voice = 'ff_siwis'
                case 'h' | 'hi' | 'hin' | 'hindi' | 'hi-in' | 'hi_in' | 'hin_deva':
                    tts_language = 'h'
                    tts_voice = 'hf_alpha'
                case 'i' | 'it' | 'ita' | 'italian' | 'it-it' | 'it_it' | 'ita_latn':
                    tts_language = 'i'
                    tts_voice = 'if_sara'
                case 'j' | 'ja' | 'jp' | 'jpn' | 'japanese' | 'ja-jp' | 'ja_jp' | 'jpn_jpan':
                    tts_language = 'j'
                    tts_voice = 'jf_gongitsune'
                case 'b' | 'br' | 'bra' | 'brazilian_portuguese' | 'portuguese' | 'pt' | 'pt-br' | 'pt_br' | 'por_latn':
                    tts_language = 'p'
                    tts_voice = 'pf_dora'
                case 'z' | 'zh' | 'zho' | 'cn' | 'chinese' | 'zh-cn' | 'zh_cn' | 'mandarin' | 'zho_hans' | 'zho_hant':
                    tts_language = 'z'
                    tts_voice = 'zf_xiaobei'
            tts.tts.set_special_setting({"language": tts_language})
            settings.SETTINGS.SetOption('tts_voice', tts_voice)

    def run_tts(self, text):
        audio_device = settings.SETTINGS.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.SETTINGS.GetOption("device_default_out_index")

        if tts.init():
            streamed_playback = settings.SETTINGS.GetOption("tts_streamed_playback")
            tts_wav = None
            if streamed_playback and hasattr(tts.tts, "tts_streaming"):
                tts_wav, sample_rate = tts.tts.tts_streaming(text)

            if tts_wav is None:
                streamed_playback = False
                tts_wav, sample_rate = tts.tts.tts(text)

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
        speaker_lang = result_obj["language"]

        # Determine target language
        target_lang = "en"
        source_lang = "en"
        if speaker_lang == self.get_plugin_setting("first_speaker_language"):
            source_lang = self.get_plugin_setting("first_source_language")
            target_lang = self.get_plugin_setting("first_target_language")
        elif speaker_lang == self.get_plugin_setting("second_speaker_language"):
            source_lang = self.get_plugin_setting("second_source_language")
            target_lang = self.get_plugin_setting("second_target_language")

        to_code = speaker_lang
        translation_text = text
        if self.get_plugin_setting("translation_enabled"):
            translation_text, from_code, to_code = texttranslate.TranslateLanguage(text, source_lang, target_lang)
            print("to_code:", to_code)

        if self.get_plugin_setting("osc_enabled"):
            osc_ip = settings.SETTINGS.GetOption("osc_ip")
            osc_port = settings.SETTINGS.GetOption("osc_port")
            osc_address = settings.SETTINGS.GetOption("osc_address")
            osc_chat_limit = settings.SETTINGS.GetOption("osc_chat_limit")
            osc_time_limit = settings.SETTINGS.GetOption("osc_time_limit")
            osc_initial_time_limit = settings.SETTINGS.GetOption("osc_initial_time_limit")
            osc_convert_ascii = settings.SETTINGS.GetOption("osc_convert_ascii")

            VRC_OSCLib.Chat_chunks(translation_text,
                                   nofify=True, address=osc_address, ip=osc_ip, port=osc_port,
                                   chunk_size=osc_chat_limit, delay=osc_time_limit,
                                   initial_delay=osc_initial_time_limit,
                                   convert_ascii=osc_convert_ascii)

        if self.get_plugin_setting("tts_enabled"):
            self.determine_tts_settings(to_code)
            self.run_tts(translation_text)

        return