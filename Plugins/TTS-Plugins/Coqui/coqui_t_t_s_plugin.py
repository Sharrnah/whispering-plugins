# ============================================================
# Coqui Text to Speech Plugin for Whispering Tiger
# V1.1.8
# Coqui: https://github.com/coqui-ai/TTS/
# Whispering Tiger: https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import base64
import io
import os
import shutil

import numpy as np
from scipy.io.wavfile import write as write_wav
from io import BytesIO

import Plugins
import audio_tools
import downloader
import settings
import websocket
import json

from pathlib import Path
import random
import subprocess
import processmanager

from functools import partial
import re
import num2words

coqui_tts_plugin_dir = Path(Path.cwd() / "Plugins" / "coqui_tts_plugin")
os.makedirs(coqui_tts_plugin_dir, exist_ok=True)

coqui_tts_tool = {
    "urls": [
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:projects/coqui-tts/coqui-tts_v1.0.1.3.zip",
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:projects/coqui-tts/coqui-tts_v1.0.1.3.zip"
    ],
    "sha256": "c6aa51e5bba4e77bcce6f85a6b535359f22c1c4262f0be3ba1b6f21f02f7dd63",
    "path": "coqui-tts",
    "version": "1013"
}

CONSTANTS = {
    "DISABLED": 'Disabled',
    "TTS": 'Text to Speech',
    "STS": 'Own Voice',
}

# for models that do not return a list of supported languages
model_languages = {
    'tts_models/multilingual/multi-dataset/xtts_v2': [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi'
    ],
    'tts_models/multilingual/multi-dataset/xtts_v1.1': [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja'
    ],
    'tts_models/multilingual/multi-dataset/xtts_v1': [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja'
    ]
}

unsupported_args = {
    "voice_dir": [
        'tts_models/multilingual/multi-dataset/xtts_v2',
        'tts_models/multilingual/multi-dataset/xtts_v1.1',
        'tts_models/multilingual/multi-dataset/xtts_v1',
    ]
}


class CoquiTTSPlugin(Plugins.Base):
    device_index = None
    download_requested = False
    websocket_connection = None
    multi_speaker = True
    multi_language = False
    process = None
    loaded_model = ""
    predefined_speakers = []

    def init(self):
        self.init_plugin_settings(
            {
                "model": {"type": "select", "value": "tts_models/en/vctk/vits",
                          "values": ['tts_models/multilingual/multi-dataset/xtts_v2', 'tts_models/multilingual/multi-dataset/xtts_v1.1', 'tts_models/multilingual/multi-dataset/xtts_v1',
                                     'tts_models/multilingual/multi-dataset/your_tts',
                                     'tts_models/multilingual/multi-dataset/bark',
                                     'tts_models/bg/cv/vits', 'tts_models/cs/cv/vits', 'tts_models/da/cv/vits',
                                     'tts_models/et/cv/vits',
                                     'tts_models/ga/cv/vits', 'tts_models/en/ek1/tacotron2',
                                     'tts_models/en/ljspeech/tacotron2-DDC',
                                     'tts_models/en/ljspeech/tacotron2-DDC_ph', 'tts_models/en/ljspeech/glow-tts',
                                     'tts_models/en/ljspeech/speedy-speech', 'tts_models/en/ljspeech/tacotron2-DCA',
                                     'tts_models/en/ljspeech/vits', 'tts_models/en/ljspeech/vits--neon',
                                     'tts_models/en/ljspeech/fast_pitch', 'tts_models/en/ljspeech/overflow',
                                     'tts_models/en/ljspeech/neural_hmm', 'tts_models/en/vctk/vits',
                                     'tts_models/en/vctk/fast_pitch',
                                     'tts_models/en/sam/tacotron-DDC', 'tts_models/en/blizzard2013/capacitron-t2-c50',
                                     'tts_models/en/blizzard2013/capacitron-t2-c150_v2',
                                     'tts_models/en/multi-dataset/tortoise-v2',
                                     'tts_models/en/jenny/jenny', 'tts_models/es/mai/tacotron2-DDC',
                                     'tts_models/es/css10/vits',
                                     'tts_models/fr/mai/tacotron2-DDC', 'tts_models/fr/css10/vits',
                                     'tts_models/uk/mai/glow-tts',
                                     'tts_models/uk/mai/vits', 'tts_models/zh-CN/baker/tacotron2-DDC-GST',
                                     'tts_models/nl/mai/tacotron2-DDC', 'tts_models/nl/css10/vits',
                                     'tts_models/de/thorsten/tacotron2-DCA', 'tts_models/de/thorsten/vits',
                                     'tts_models/de/thorsten/tacotron2-DDC', 'tts_models/de/css10/vits-neon',
                                     'tts_models/ja/kokoro/tacotron2-DDC', 'tts_models/tr/common-voice/glow-tts',
                                     'tts_models/it/mai_female/glow-tts', 'tts_models/it/mai_female/vits',
                                     'tts_models/it/mai_male/glow-tts', 'tts_models/it/mai_male/vits',
                                     'tts_models/ewe/openbible/vits',
                                     'tts_models/hau/openbible/vits', 'tts_models/lin/openbible/vits',
                                     'tts_models/tw_akuapem/openbible/vits', 'tts_models/tw_asante/openbible/vits',
                                     'tts_models/yor/openbible/vits', 'tts_models/hu/css10/vits',
                                     'tts_models/el/cv/vits',
                                     'tts_models/fi/css10/vits', 'tts_models/hr/cv/vits', 'tts_models/lt/cv/vits',
                                     'tts_models/lv/cv/vits', 'tts_models/mt/cv/vits', 'tts_models/pl/mai_female/vits',
                                     'tts_models/pt/cv/vits', 'tts_models/ro/cv/vits', 'tts_models/sk/cv/vits',
                                     'tts_models/sl/cv/vits',
                                     'tts_models/sv/cv/vits', 'tts_models/ca/custom/vits',
                                     'tts_models/fa/custom/glow-tts',
                                     'tts_models/bn/custom/vits-male', 'tts_models/bn/custom/vits-female',
                                     'tts_models/???/fairseq/vits']},
                "use_gpu": True,

                "speaker_overwrite": "",
                "language": {"type": "select", "value": "", "values": ['']},
                "language_overwrite": "",
                "websocket_ip": "127.0.0.1",
                "websocket_port": "5000",
                "sample_rate_overwrite": "",

                "fairseq_language": "eng",
                "zzz_fairseq_info": {
                    "label": "Fairseq supports ~1100 languages. Please set model to fairseq and enter the language ISO code which you can find under the link:",
                    "type": "label", "style": "left"},
                "zzzz_fairseq_iso_codes": {"label": "Open ISO code list",
                                           "value": "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:share/fairseq-tts-supported-languages.html",
                                           "type": "hyperlink"},

                "api_key": "",
                "emotion": {"type": "select", "value": "", "values":
                    ["", "Neutral", "Happy", "Sad", "Angry", "Dull"]
                            },
                "speed": {"type": "slider", "min": 0.0, "max": 2.0, "step": 0.1, "value": 0.0},

                "model_load_btn": {"label": "Load model", "type": "button", "style": "primary"},
                "custom_model_file": {"type": "file_open", "accept": ".pth", "value": ""},
                # "model_config_file": {"type": "file_open", "accept": ".json", "value": ""},
                "speakers_folder": {"type": "folder_open", "accept": "", "value": ""},
                # "speakers_file_path": {"type": "file_open", "accept": ".pth", "value": ""},

                "voice_change_source": {"type": "select", "value": CONSTANTS["DISABLED"],
                                        "values": [CONSTANTS["TTS"], CONSTANTS["STS"], CONSTANTS["DISABLED"]]},
                "voice_change_clone_target": {"type": "file_open", "accept": ".wav", "value": ""},

                "replace_numbers_to_words": False,
                "numbers_language": {"type": "select", "value": "en",
                                     "values": list(num2words.CONVERTER_CLASSES.keys())},
                "replace_abbreviations": False,
            },
            settings_groups={
                "General": ["speaker_overwrite", "language", "language_overwrite", "sample_rate_overwrite"],
                "Server": ["websocket_ip", "websocket_port"],
                "Fairseq Model": ["fairseq_language", "zzz_fairseq_info", "zzzz_fairseq_iso_codes"],
                # "Coqui Studio": ["api_key", "emotion", "speed"],
                "Model": ["use_gpu", "model", "custom_model_file", "speakers_folder", "model_load_btn"],
                "Voice Change": ["voice_change_source", "voice_change_clone_target"],
                "Text Preprocessing": ["replace_numbers_to_words", "replace_abbreviations", "numbers_language"],
            }
        )

        if self.is_enabled(False):
            # check version from VERSION file
            version_file = Path(coqui_tts_plugin_dir / "VERSION")
            if version_file.is_file():
                version = version_file.read_text().strip()
                if version != coqui_tts_tool["version"] and Path(
                        coqui_tts_plugin_dir / coqui_tts_tool["path"]).is_dir():
                    print("coqui-tts tool VERSION file is not " + coqui_tts_tool["version"] + ". removing old version.")
                    # delete old version folder
                    shutil.rmtree(str(Path(coqui_tts_plugin_dir / coqui_tts_tool["path"]).resolve()))

            # download coqui-tts
            if not Path(coqui_tts_plugin_dir / coqui_tts_tool["path"] / "coqui-tts.exe").is_file():
                if Path(coqui_tts_plugin_dir / coqui_tts_tool["path"]).is_dir():
                    print("removing old version...")
                    shutil.rmtree(str(Path(coqui_tts_plugin_dir / coqui_tts_tool["path"]).resolve()))
                print("coqui-tts tool downloading...")
                # download from random url in list
                voice_clone_url = random.choice(coqui_tts_tool["urls"])
                downloader.download_extract([voice_clone_url],
                                            str(coqui_tts_plugin_dir.resolve()),
                                            coqui_tts_tool["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(coqui_tts_plugin_dir / os.path.basename(voice_clone_url)),
                                                str(coqui_tts_plugin_dir.resolve()),
                                            ),
                                            title="Coqui TTS - app", extract_format="zip")

            # disable default tts engine
            settings.SetOption("tts_enabled", False)

            if Path(coqui_tts_plugin_dir / coqui_tts_tool["path"] / "coqui-tts.exe").is_file():
                self.run_process()

        pass

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        if self.process is not None:
            processmanager.kill_process(self.process)
        pass

    def is_inside_xml_tag(self, match, text):
        open_tag_pos = text.rfind('<', 0, match.start())
        close_tag_pos = text.rfind('>', 0, match.start())
        return open_tag_pos > close_tag_pos

    def replace_numbers(self, match, lang, text):
        if self.is_inside_xml_tag(match, text):
            return match.group(0)
        else:
            return num2words.num2words(int(match.group(0)), lang=lang)

    def replace_special_characters_per_model(self, text):
        # fix for german tts_models that do not support special characters.
        models = ["tts_models/de/thorsten/tacotron2-DCA", "tts_models/de/thorsten/tacotron2-DDC"]
        if self.loaded_model in models:
            if 'ß' in text or 'ẞ' in text:
                # handle the 'ß' following an uppercase character
                text = ''.join(['SS' if text[i] == 'ß' and text[i-1].isupper() else text[i] for i in range(len(text))])

            trans_lower = str.maketrans({
                'ä': 'ae',
                'ö': 'oe',
                'ü': 'ue',
                'ß': 'ss',
            })
            trans_upper = str.maketrans({
                'Ä': 'AE',
                'Ö': 'OE',
                'Ü': 'UE',
            })
            text = text.translate(trans_lower).translate(trans_upper)

        # fix for tts_models/de/thorsten/tacotron2-DCA that repeats last word if sentence does not end with punctuation
        models = ["tts_models/de/thorsten/tacotron2-DCA"]
        if self.loaded_model in models:
            if not text.endswith(".") and not text.endswith("!") and not text.endswith("?") and not text.endswith(
                    ",") and not text.endswith(";") and not text.endswith(":") and not text.endswith(
                    ")") and not text.endswith("]"):
                text += "."
        return text

    def run_process(self):
        # run command line tool with parameters
        try:
            model = self.get_plugin_setting("model")
            if self.get_plugin_setting("custom_model_file") != "":
                model = self.get_plugin_setting("custom_model_file")

            fairseq_language = self.get_plugin_setting("fairseq_language")
            websocket_ip = self.get_plugin_setting("websocket_ip")
            websocket_port = str(self.get_plugin_setting("websocket_port"))
            use_gpu = self.get_plugin_setting("use_gpu")

            self.loaded_model = model

            process_arguments = [str(Path(coqui_tts_plugin_dir / coqui_tts_tool["path"] / "coqui-tts.exe").resolve()),
                                 "--model_name",
                                 model, "--fairseq_lang", fairseq_language, "--websocket_ip", websocket_ip,
                                 "--websocket_port", websocket_port]
            if use_gpu:
                process_arguments.append("--use_gpu")

            if self.process is not None:
                processmanager.kill_process(self.process)

            self.process = processmanager.run_process(process_arguments, env={
                'TTS_HOME': str(coqui_tts_plugin_dir.resolve()),
                'COQUI_TOS_AGREED': '1',
                'PYTHONIOENCODING': ':replace',
            })

            # if model is xtts_v2, send studio speakers list
            if model.endswith("xtts_v2"):
                self.predefined_speakers = ['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']

        except subprocess.CalledProcessError as e:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": "Running Coqui TTS Server failed.: " + str(e)}))
            return

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

        audio_tools.play_audio(wav, audio_device,
                               source_sample_rate=source_sample_rate,
                               audio_device_channel_num=audio_device_channel_num,
                               target_channels=target_channels,
                               input_channels=input_channels,
                               dtype=dtype,
                               secondary_device=secondary_audio_device, tag="tts")

    def get_plugin(self, class_name):
        for plugin_inst in Plugins.plugins:
            if plugin_inst.__class__.__name__ == class_name:
                return plugin_inst  # return plugin instance
        return None

    def numpy_array_to_wav_bytes(self, audio: np.ndarray, sample_rate: int = 22050) -> BytesIO:
        buff = io.BytesIO()
        write_wav(buff, sample_rate, audio)
        buff.seek(0)
        return buff


    def on_event_received(self, message, websocket_connection=None):
        if self.is_enabled(False):
            if "type" not in message:
                return
            if message["type"] == "plugin_button_press":
                if message["value"] == "model_load_btn":
                    # restart coqui application
                    self.run_process()
                    pass

            # custom events
            if message["type"] == "plugin_custom_event":
                if "wav_data" in message:
                    wav_data = message["wav_data"]
                    sample_rate = message["sample_rate"]
                    if self.get_plugin_setting("sample_rate_overwrite") != "":
                        sample_rate = int(self.get_plugin_setting("sample_rate_overwrite"))

                    # base664decode wav_data
                    wav = base64.b64decode(wav_data)

                    # call custom plugin event method
                    plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': wav, 'sample_rate': sample_rate})
                    if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
                        wav = plugin_audio['audio']
                        wav_data = base64.b64encode(wav).decode('utf-8')

                    if not self.download_requested:
                        self.play_audio_on_device(wav, self.device_index,
                                                  source_sample_rate=sample_rate,
                                                  audio_device_channel_num=2,
                                                  target_channels=2,
                                                  input_channels=1,
                                                  dtype="int16"
                                                  )
                    else:
                        if self.websocket_connection is not None:
                            websocket.AnswerMessage(self.websocket_connection,
                                                    json.dumps({"type": "tts_save", "wav_data": wav_data}))
                elif "speakers" in message:
                    print("speakers", message["speakers"])
                    if len(message["speakers"]) == 0:
                        speakers_list = [""]
                        self.multi_speaker = False
                    else:
                        speakers_list = [""]
                        speakers_list.extend(message["speakers"])
                        self.multi_speaker = True

                    #special case for XTTS v2 and its studio speakers
                    if len(self.predefined_speakers) > 0:
                        speakers_list = [""]
                        speakers_list.extend(self.predefined_speakers)
                        self.multi_speaker = True

                    websocket.BroadcastMessage(json.dumps({
                        "type": "available_tts_voices",
                        "data": speakers_list
                    }), exclude_client=websocket_connection)

                elif "languages" in message:
                    print("languages", message["languages"])
                    if len(message["languages"]) == 0:
                        languages_list = [""]
                        self.multi_language = False
                        if self.loaded_model in model_languages:
                            languages_list.extend(model_languages[self.loaded_model])
                            self.multi_language = True
                    else:
                        languages_list = [""]
                        languages_list.extend(message["languages"])
                        self.multi_language = True
                    self.set_plugin_setting("language", {"type": "select", "value": self.get_plugin_setting("language"),
                                                         "values": languages_list})

                elif "error" in message:
                    print("received error:", message["error"])
                    websocket.BroadcastMessage(json.dumps({
                        "type": "error",
                        "data": message["error"]
                    }), exclude_client=websocket_connection)

                else:
                    return

                # reset values
                self.websocket_connection = None
                self.download_requested = False
        else:
            websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Plugin is disabled."}))

    def request_tts_data(self, text, device_index, websocket_connection=None, download=False, speech_data=None,
                         sample_rate=None):
        self.device_index = device_index
        self.download_requested = download
        self.websocket_connection = websocket_connection

        model = self.get_plugin_setting("model")

        clone_voice = ""
        if self.get_plugin_setting("voice_change_source") != CONSTANTS["DISABLED"]:
            clone_voice = self.get_plugin_setting("voice_change_clone_target")
        speaker = settings.GetOption("tts_voice")
        if not self.multi_speaker:
            speaker = ""

        # special case for XTTS v2 and its studio speakers
        if model.endswith("xtts_v2") and settings.GetOption("tts_voice") in self.predefined_speakers:
            speaker = settings.GetOption("tts_voice")
        if self.get_plugin_setting("speaker_overwrite") != "":
            speaker = self.get_plugin_setting("speaker_overwrite")

        language = self.get_plugin_setting("language")
        if not self.multi_language:
            language = ""
        if self.get_plugin_setting("language_overwrite") != "" and self.get_plugin_setting("language_overwrite") is not None:
            language = self.get_plugin_setting("language_overwrite")

        speakers_folder = self.get_plugin_setting("speakers_folder")
        if speakers_folder == "":
            speakers_folder = None
        if speakers_folder is not None:
            if self.loaded_model in unsupported_args["voice_dir"]:
                speakers_folder = None

        # text preprocessing
        numbers_to_words = self.get_plugin_setting("replace_numbers_to_words")
        numbers_lang = self.get_plugin_setting("numbers_language")
        replace_abbreviations = self.get_plugin_setting("replace_abbreviations")

        emotion = self.get_plugin_setting("emotion")
        speed = self.get_plugin_setting("speed")
        if speed == 0.0:
            speed = None
        if self.get_plugin_setting("api_key") == "":
            emotion = None
            speed = None

        if speech_data is None and text.strip() != "":
            # replace all numbers with their word representations
            if numbers_to_words:
                replace_numbers_with_lang = partial(self.replace_numbers, lang=numbers_lang, text=text)
                text = re.sub(r"\d+", replace_numbers_with_lang, text)

            text = self.replace_special_characters_per_model(text)

            websocket.BroadcastMessage(json.dumps({"type": "coqui_generate_wav_data", "data": {
                "text": text, "replace_abbreviations": replace_abbreviations, "clone_wav": clone_voice,
                "emotion": emotion, "speed": speed, "speaker": speaker,
                "language": language, "voice_dir": speakers_folder,
            }}))
        elif speech_data is not None:
            # wav_bytes = numpy_array_to_wav_bytes(wav_data, SAMPLE_RATE)
            speech_data = base64.b64encode(speech_data).decode('utf-8')

            websocket.BroadcastMessage(json.dumps({"type": "coqui_apply_voice_change", "data": {
                "wav_data": speech_data, "clone_wav": clone_voice, "sample_rate": sample_rate
            }}))

    def stt(self, text, result_obj):
        if not self.is_enabled(False):
            return
        if self.get_plugin_setting("voice_change_source") == CONSTANTS["STS"]:
            return
        if self.is_enabled(False) and settings.GetOption("tts_answer") and text.strip() != "":
            device_index = settings.GetOption("device_out_index")
            if device_index is None or device_index == -1:
                device_index = settings.GetOption("device_default_out_index")

            self.request_tts_data(text, device_index, None, False)
        pass

    def tts(self, text, device_index, websocket_connection=None, download=False):
        if not self.is_enabled(False):
            return
        if self.get_plugin_setting("voice_change_source") == CONSTANTS["STS"]:
            return
        if device_index is None or device_index == -1:
            device_index = settings.GetOption("device_default_out_index")

        self.request_tts_data(text, device_index, websocket_connection, download)
        pass

    def sts(self, wavefiledata, sample_rate):
        if not self.is_enabled(False) or self.get_plugin_setting("voice_change_source") != CONSTANTS["STS"]:
            return
        voice_change_clone_target = self.get_plugin_setting("voice_change_clone_target")
        if not voice_change_clone_target or wavefiledata is None:
            websocket.BroadcastMessage(json.dumps({"type": "info",
                                                   "data": "No clone target audio file found. Please select a clone audio file first (between 4 - 8 seconds."}))
            return

        device_index = settings.GetOption("device_out_index")
        if device_index is None or device_index == -1:
            device_index = settings.GetOption("device_default_out_index")

        self.request_tts_data("", device_index, None, False, wavefiledata, sample_rate)
