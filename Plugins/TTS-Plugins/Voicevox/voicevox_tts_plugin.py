# ============================================================
# Voicevox Text to Speech Plugin for Whispering Tiger
# V1.2.6
# See https://github.com/Sharrnah/whispering
# ============================================================
#
import asyncio
import base64
import json
import sys
import threading
from importlib import util

import Plugins

import numpy as np

from pathlib import Path
import os

import audio_tools
import settings
import websocket
import downloader
import shutil


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


voicevox_plugin_dir = Path(Path.cwd() / "Plugins" / "voicevox_plugin")
os.makedirs(voicevox_plugin_dir, exist_ok=True)

voicevox_core_python_repository = {
    "CPU": {
        "url": "https://github.com/VOICEVOX/voicevox_core/releases/download/0.15.0-preview.15/voicevox_core-0.15.0rc15+cpu-cp38-abi3-win_amd64.whl",
        "sha256": "8499f9c6f044f9fee9d1431e1ba9780026d89f09c018fb322b85be252aa2d299"
    },
    "CUDA": {
        "url": "https://github.com/VOICEVOX/voicevox_core/releases/download/0.15.0-preview.15/voicevox_core-0.15.0rc15+cuda-cp38-abi3-win_amd64.whl",
        "sha256": "2608d8a48a07687a775a225d7963e9b4a6f06327542124545aa0fe565985f237"
    },
    "DIRECTML": {
        "url": "https://github.com/VOICEVOX/voicevox_core/releases/download/0.15.0-preview.15/voicevox_core-0.15.0rc15+directml-cp38-abi3-win_amd64.whl",
        "sha256": "c8c64c583163bff2da7a0c8289f0e384dc9f30a7b4262bd9de0e619b1307712e"
    },
    "version": "0.15.0-preview.15"
}
voicevox_core_dll_repository = {
    "CPU": {
        "url": "https://github.com/VOICEVOX/voicevox_core/releases/download/0.15.0-preview.15/voicevox_core-windows-x64-cpu-0.15.0-preview.15.zip",
        "sha256": "45423b438ad1141095211abaf1fa6bfeeb6e9b7fc37f5796fff2f3902819e2c9",
        "path": "voicevox_core-windows-x64-cpu-0.15.0-preview.15"
    },
    "CUDA": {
        "url": "https://github.com/VOICEVOX/voicevox_core/releases/download/0.15.0-preview.15/voicevox_core-windows-x64-cuda-0.15.0-preview.15.zip",
        "sha256": "60c11754eccfbadb366397c4b75c603ad42aa2d193d7af4074786a4cf16deeb2",
        "path": "voicevox_core-windows-x64-cuda-0.15.0-preview.15"
    },
    "DIRECTML": {
        "url": "https://github.com/VOICEVOX/voicevox_core/releases/download/0.15.0-preview.15/voicevox_core-windows-x64-directml-0.15.0-preview.15.zip",
        "sha256": "633ff1ecc9cd20be3ff6dd9c941950ad30ee6dce446356cfc8d832271806ab82",
        "path": "voicevox_core-windows-x64-directml-0.15.0-preview.15"
    },
    "version": "0.15.0-preview.15"
}
voicevox_models = {
    "url": "https://github.com/VOICEVOX/voicevox_core/releases/download/0.15.0-preview.15/model-0.15.0-preview.15.zip",
    "sha256": "f7256dc5a5a8387ca1d29b22695afcb2783e7de46918b10b6b306b72e51446aa",
    "path": "model-0.15.0-preview.15",
    "version": "0.15.0-preview.15",
}

open_jtalk_dict_file = {
    "url": "https://jaist.dl.sourceforge.net/project/open-jtalk/Dictionary/open_jtalk_dic-1.11/open_jtalk_dic_utf_8-1.11.tar.gz",
    "sha256": "33e9cd251bc41aa2bd7ca36f57abbf61eae3543ca25ca892ae345e394cb10549",
    "path": "open_jtalk_dic_utf_8-1.11",
    "version": "1.11"
}

pydantic_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/8a/64/db1aafc37fab0dad89e0a27f120a18f2316fca704e9f95096ade47b933ac/pydantic-1.10.7-cp310-cp310-win_amd64.whl",
    "sha256": "a7cd2251439988b413cb0a985c4ed82b6c6aac382dbaff53ae03c4b23a70e80a",
    "path": "pydantic",
    "version": "1.10.7"
}


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


def run_async_function_in_thread(async_func):
    result = None
    exception = None

    def thread_func():
        nonlocal result, exception

        async def nested_async():
            nonlocal result, exception
            try:
                result = await async_func()
            except Exception as e:
                exception = e

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(nested_async())
        finally:
            loop.close()

    thread = threading.Thread(target=thread_func)
    thread.start()
    thread.join()

    if exception:
        raise exception
    return result


class VoicevoxTTSPlugin(Plugins.Base):
    core = None
    synthesizer = None
    sample_rate = 24000
    acceleration_mode = "CPU"
    voicevox_core_module = None
    previous_model = None
    open_jtalk_dict_path = None
    model = None

    speakers = []

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "model": {"type": "select", "value": "0", "values": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]},
                "model_load_btn": {"label": "Load model", "type": "button", "style": "primary"},

                #"speaker_list_link": {"label": "Open Speaker List", "value": "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:share/voicevox-voice-ids.html", "type": "hyperlink"},
                "acceleration_mode": {"type": "select", "value": "CPU", "values": ["CPU", "CUDA", "DIRECTML"]},

                "speed_scale": 1.0,
                "volume_scale": 1.0,
                "intonation_scale": 1.0,
                "pre_phoneme_length": 0.0,
                "post_phoneme_length": 0.0
            },
            settings_groups={
                "General": ["model", "model_load_btn", "acceleration_mode"],
                "Settings": ["speed_scale", "volume_scale", "intonation_scale", "pre_phoneme_length", "post_phoneme_length"],
            }
        )

        if self.is_enabled(False):
            # disable default tts engine
            settings.SetOption("tts_enabled", False)

            self.acceleration_mode = self.get_plugin_setting("acceleration_mode", "CPU")

            os.makedirs(Path(voicevox_plugin_dir / self.acceleration_mode), exist_ok=True)

            websocket.set_loading_state("voicevox_plugin_loading", True)

            needs_update = should_update_version_file_check(
                Path(voicevox_plugin_dir / self.acceleration_mode / "voicevox_core"),
                voicevox_core_dll_repository["version"]
            )
            if needs_update and Path(voicevox_plugin_dir / self.acceleration_mode / "voicevox_core").is_dir():
                print("Removing old voicevox_core directory")
                shutil.rmtree(str(Path(voicevox_plugin_dir / self.acceleration_mode / "voicevox_core").resolve()))
            if not Path(voicevox_plugin_dir / self.acceleration_mode / "voicevox_core" / "__init__.py").is_file() or needs_update:
                downloader.download_extract([voicevox_core_python_repository[self.acceleration_mode]["url"]],
                                            str(Path(voicevox_plugin_dir / self.acceleration_mode).resolve()),
                                            voicevox_core_python_repository[self.acceleration_mode]["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(Path(voicevox_plugin_dir / self.acceleration_mode / os.path.basename(voicevox_core_python_repository[self.acceleration_mode]["url"])).resolve()),
                                                str(Path(voicevox_plugin_dir / self.acceleration_mode).resolve()),
                                            ),
                                            title="Voicevox Core", extract_format="zip")

            if not Path(voicevox_plugin_dir / voicevox_models["path"]).is_dir() or needs_update:
                downloader.download_extract([voicevox_models["url"]],
                                            str(Path(voicevox_plugin_dir).resolve()),
                                            voicevox_models["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(Path(voicevox_plugin_dir / os.path.basename(voicevox_models["url"])).resolve()),
                                                str(Path(voicevox_plugin_dir).resolve()),
                                            ),
                                            title="Voicevox Models", extract_format="zip")

            if not Path(voicevox_plugin_dir / self.acceleration_mode / "voicevox_core" / "voicevox_core.lib").is_file() or needs_update:
                downloader.download_extract([voicevox_core_dll_repository[self.acceleration_mode]["url"]],
                                            str(Path(voicevox_plugin_dir / self.acceleration_mode).resolve()),
                                            voicevox_core_dll_repository[self.acceleration_mode]["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(Path(voicevox_plugin_dir / self.acceleration_mode / os.path.basename(voicevox_core_dll_repository[self.acceleration_mode]["url"]))),
                                                str(Path(voicevox_plugin_dir / self.acceleration_mode).resolve()),
                                            ),
                                            title="Voicevox Core lib")
                # move dll files to voicevox_core directory
                downloader.move_files(str(Path(voicevox_plugin_dir / self.acceleration_mode / voicevox_core_dll_repository[self.acceleration_mode]["path"]).resolve()), str(Path(voicevox_plugin_dir / self.acceleration_mode / "voicevox_core").resolve()))
                # # move vvm model files to voicevox_core directory
                # os.makedirs(Path(voicevox_plugin_dir / self.acceleration_mode / "voicevox_core" / "model"), exist_ok=True)
                # downloader.move_files(str(Path(voicevox_plugin_dir / self.acceleration_mode / voicevox_core_dll_repository[self.acceleration_mode]["path"] / "model").resolve()), str(Path(voicevox_plugin_dir / self.acceleration_mode / "voicevox_core" / "model").resolve()))
                # delete obsolete dll folder
                shutil.rmtree(Path(voicevox_plugin_dir / self.acceleration_mode / voicevox_core_dll_repository[self.acceleration_mode]["path"]))
                # write version file
                write_version_file(
                    Path(voicevox_plugin_dir / self.acceleration_mode / "voicevox_core"),
                    voicevox_core_dll_repository["version"]
                )

            self.open_jtalk_dict_path = Path(voicevox_plugin_dir / open_jtalk_dict_file["path"])
            needs_update = should_update_version_file_check(
                self.open_jtalk_dict_path,
                open_jtalk_dict_file["version"]
            )
            if not Path(self.open_jtalk_dict_path / "sys.dic").is_file() or needs_update:
                if self.open_jtalk_dict_path.is_dir():
                    print("Removing old Open JTalk dictionary directory")
                    shutil.rmtree(str(self.open_jtalk_dict_path.resolve()))
                downloader.download_extract([open_jtalk_dict_file["url"]],
                                            str(voicevox_plugin_dir.resolve()),
                                            open_jtalk_dict_file["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_tar_gz,
                                            fallback_extract_func_args=(
                                                str(voicevox_plugin_dir / os.path.basename(open_jtalk_dict_file["url"])),
                                                str(voicevox_plugin_dir.resolve()),
                                            ),
                                            title="Open JTalk dictionary")
                # write version file
                write_version_file(
                    self.open_jtalk_dict_path,
                    open_jtalk_dict_file["version"]
                )

            # load the pydantic module
            needs_update = should_update_version_file_check(
                Path(voicevox_plugin_dir / "pydantic"),
                pydantic_dependency_module["version"]
            )
            if not Path(voicevox_plugin_dir / "pydantic" / "__init__.py").is_file() or needs_update:
                if Path(voicevox_plugin_dir / "pydantic").is_dir():
                    print("Removing old Pydantic module directory")
                    shutil.rmtree(str(Path(voicevox_plugin_dir / "pydantic").resolve()))
                downloader.download_extract([pydantic_dependency_module["url"]],
                                            str(voicevox_plugin_dir.resolve()),
                                            pydantic_dependency_module["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(voicevox_plugin_dir / os.path.basename(pydantic_dependency_module["url"])),
                                                str(voicevox_plugin_dir.resolve()),
                                            ),
                                            title="Pydantic", extract_format="zip")
                # write version file
                write_version_file(
                    Path(voicevox_plugin_dir / "pydantic"),
                    pydantic_dependency_module["version"]
                )

            print("loading Pydantic module...")
            pydantic = load_module(str(Path(voicevox_plugin_dir / pydantic_dependency_module["path"]).resolve()))

            # load the voicevox_core module
            if self.voicevox_core_module is None:
                self.voicevox_core_module = load_module(str(Path(voicevox_plugin_dir / self.acceleration_mode / "voicevox_core").resolve()))

            if self.synthesizer is None:
                self.load_model(self.get_plugin_setting("model"))
            websocket.set_loading_state("voicevox_plugin_loading", False)
        pass

    def get_style_names(self, speakers):
        """Get a list of formatted strings combining speaker names with style names."""
        style_names = []
        for speaker in speakers:
            for style in speaker.styles:
                style_names.append(f"{speaker.name} - {style.name}")
        return style_names

    def get_style_id(self, speakers, combined_style):
        """Get the ID of a style based on a combined string of speaker and style names."""
        speaker_name, style_name = combined_style.split(" - ")
        for speaker in speakers:
            if speaker.name == speaker_name:
                for style in speaker.styles:
                    if style.name == style_name:
                        return style.id
        return None

    def load_model(self, model_name):
        if self.previous_model != model_name:
            websocket.set_loading_state("voicevox_model_loading", True)
            if self.synthesizer is not None and self.model is not None and self.synthesizer.is_loaded_voice_model(self.model.id):
                self.synthesizer.unload_voice_model(self.model.id)

            acceleration_mode = "AUTO"
            if self.acceleration_mode == "CPU":
                acceleration_mode = self.voicevox_core_module.AccelerationMode.CPU
            elif self.acceleration_mode == "CUDA" or self.acceleration_mode == "GPU":
                acceleration_mode = self.voicevox_core_module.AccelerationMode.GPU

            load_all_models = False
            if model_name == "All":
                load_all_models = True

            print("loading synthesizer...")
            self.synthesizer = self.voicevox_core_module.Synthesizer(
                self.voicevox_core_module.OpenJtalk(str(self.open_jtalk_dict_path.resolve())), acceleration_mode=acceleration_mode
            )

            vvm_path = Path(voicevox_plugin_dir / voicevox_models["path"] / (model_name + ".vvm"))
            print("init voice model...")
            self.model = run_async_function_in_thread(lambda: self.voicevox_core_module.VoiceModel.from_path(vvm_path))
            print("loading voice model...")
            run_async_function_in_thread(lambda: self.synthesizer.load_voice_model(self.model))
            print("loading voice model finished...")

            websocket.set_loading_state("voicevox_model_loading", False)
        self.previous_model = model_name

        self.speakers = self.synthesizer.metas

        websocket.BroadcastMessage(json.dumps({
            "type": "available_tts_voices",
            "data": self.get_style_names(self.speakers)
        }))

    def apply_rvc(self, buff):
        # call custom plugin event method
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': buff, 'sample_rate': self.sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            buff = plugin_audio['audio']
        return buff

    def predict(self, text, speaker):
        speed_scale = self.get_plugin_setting("speed_scale", 1.0)
        volume_scale = self.get_plugin_setting("volume_scale", 1.0)
        intonation_scale = self.get_plugin_setting("intonation_scale", 1.0)
        pre_phoneme_length = self.get_plugin_setting("pre_phoneme_length", 0.0)
        post_phoneme_length = self.get_plugin_setting("post_phoneme_length", 0.0)

        if len(text.strip()) == 0:
            return np.zeros(0).astype(np.int16)

        #audio_query = self.core.audio_query(text, speaker)
        audio_query = run_async_function_in_thread(lambda: self.synthesizer.audio_query(text, speaker))

        audio_query.speed_scale = speed_scale
        audio_query.volume_scale = volume_scale
        audio_query.intonation_scale = intonation_scale
        audio_query.pre_phoneme_length = pre_phoneme_length
        audio_query.post_phoneme_length = post_phoneme_length

        wav = run_async_function_in_thread(lambda: self.synthesizer.synthesis(audio_query, speaker))

        return wav

    def generate_tts(self, text):
        combined_style = settings.GetOption("tts_voice")
        speaker = None
        if combined_style:
            speaker = self.get_style_id(self.speakers, combined_style)

        if speaker is None:
            websocket.BroadcastMessage(json.dumps({"type": "info",
                                                   "data": "No speaker selected. Please select a speaker from the list in the Text-to-Speech tab."}))
            return None

        wav = self.predict(text, speaker)

        wav = self.apply_rvc(wav)

        return wav

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
                               input_channels=1,
                               dtype=dtype,
                               secondary_device=secondary_audio_device, tag="tts")

    def stt(self, text, result_obj):
        if self.is_enabled(False) and settings.GetOption("tts_answer") and text.strip() != "":
            audio_device = settings.GetOption("device_out_index")
            if audio_device is None or audio_device == -1:
                audio_device = settings.GetOption("device_default_out_index")
            wav = self.generate_tts(text.strip())
            if wav is not None:
                self.play_audio_on_device(wav, audio_device)
        return

    def tts(self, text, device_index, websocket_connection=None, download=False, path=''):
        if self.is_enabled(False):
            if device_index is None or device_index == -1:
                device_index = settings.GetOption("device_default_out_index")

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
                    self.play_audio_on_device(wav, device_index)
        return

    def on_event_received(self, message, websocket_connection=None):
        if self.is_enabled(False):
            if "type" not in message:
                return
            if message["type"] == "plugin_button_press":
                if message["value"] == "model_load_btn":
                    self.load_model(self.get_plugin_setting("model"))
                    pass

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        pass
