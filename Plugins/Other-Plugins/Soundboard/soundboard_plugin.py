# ============================================================
# Soundboard Plugin for Whispering Tiger
# V0.0.3
# See https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import io
import json
import os

from pydub import AudioSegment
from pathlib import Path

import soundfile

import Plugins
import audio_tools
import downloader
import settings
import websocket

soundboard_plugin_dir = Path(Path.cwd() / "Plugins" / "soundboard_plugin")
os.makedirs(soundboard_plugin_dir, exist_ok=True)

soundboard_plugin_sounds_dir = Path(soundboard_plugin_dir / "sounds")
os.makedirs(soundboard_plugin_sounds_dir, exist_ok=True)

sound_packs = {
    "pack 1": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:projects/soundboard_plugin/soundboard_sounds1.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:projects/soundboard_plugin/soundboard_sounds1.zip"
        ],
        "sha256": "27cafa41b6fa37480ded64305a87309348618847750aafa750e64aaed0e8e8ca",
    }
}


class SoundboardPlugin(Plugins.Base):
    target_sample_rate = 44000

    def init(self):
        sounds_folder = str(soundboard_plugin_sounds_dir.resolve())

        grouped_sounds_buttons = {}
        sounds_buttons = {}

        if self.is_enabled(False):
            # read soundboard_folder.txt file
            try:
                with open(str(Path(soundboard_plugin_dir / "soundboard_folder.txt").resolve()), "r") as f:
                    sounds_folder = f.read()
            except:
                print("Could not read soundboard_folder.txt file.")

            # dictionary to group sounds by their subfolder
            if sounds_folder != "":
                for root, dirs, files in os.walk(sounds_folder):
                    for name in files:
                        btn_name = "sound_play_btn_" + name
                        original_group_name = os.path.relpath(root, sounds_folder)
                        display_group_name = "General" if original_group_name == "." else original_group_name.capitalize()

                        if name.endswith(".wav"):
                            grouped_sounds_buttons.setdefault(display_group_name, []).append((btn_name, original_group_name))
                        elif name.endswith(".mp3"):
                            sound_mp3 = os.path.join(root, name)
                            name_wav = name.replace(".mp3", ".wav")
                            sound_wav = os.path.join(root, name_wav)
                            if not os.path.exists(sound_wav):
                                print(f"Could not find wav file: {sound_wav} - converting from mp3.")
                                self.convert_mp3_to_wav(sound_mp3, sound_wav)
                                grouped_sounds_buttons.setdefault(display_group_name, []).append(("sound_play_btn_" + name_wav, original_group_name))
                        else:
                            continue

            # flatten the dictionary to create all buttons
            for display_group, btn_data in grouped_sounds_buttons.items():
                for btn_name, original_group_name in btn_data:
                    label = btn_name.replace("sound_play_btn_", "").replace(".wav", "")
                    value = os.path.join(sounds_folder, original_group_name, label + ".wav")
                    sounds_buttons[btn_name] = {
                        "label": label, "type": "button", "style": "primary", "value": value
                    }

        # prepare all possible settings
        self.init_plugin_settings(
            {
                # General
                **sounds_buttons,

                # Manage
                "sounds_folder": {"type": "folder_open", "accept": "", "value": sounds_folder},
                "sounds_folder_save_button": {"label": "Save", "type": "button", "style": "primary"},
                "stop_playing": {"label": "Stop playing", "type": "button", "style": "primary"},
                "allow_overlapping_audio": False,
                "download_soundpack_button": {"label": "Download Demo Soundpack (also resets to default sounds_folder)", "type": "button", "style": "primary", "value": "pack 1"},
            },
            settings_groups={
                **{group: [btn_data[0] for btn_data in btn_list] for group, btn_list in grouped_sounds_buttons.items()},
                "zz_Manage": ["sounds_folder", "sounds_folder_save_button", "stop_playing", "allow_overlapping_audio"],
                "zz_Soundpacks": ["download_soundpack_button"]
            }
        )

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
                               stop_play=stop_play, tag="soundboard")

    def on_event_received(self, message, websocket_connection=None):
        if self.is_enabled(False):
            if "type" not in message:
                return
            if message["type"] == "plugin_button_press":
                if message["value"] == "sounds_folder_save_button":
                    # write txt file with folder path
                    sounds_folder = self.get_plugin_setting("sounds_folder")
                    if sounds_folder != "":
                        # write text file
                        with open(str(Path(soundboard_plugin_dir / "soundboard_folder.txt").resolve()), "w") as f:
                            f.write(sounds_folder)
                        self.init()
                        websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Saved sounds folder.\nMake sure it contains your sounds."}))
                    pass
                if message["value"] == "stop_playing":
                    audio_tools.stop_audio(tag="soundboard")
                if message["value"] == "download_soundpack_button":
                    soundpack_name = self.get_plugin_setting(message["value"])
                    soundpack_urls = sound_packs[soundpack_name]["urls"]
                    downloader.download_extract(soundpack_urls,
                                                str(soundboard_plugin_sounds_dir.resolve()),
                                                sound_packs[soundpack_name]["sha256"],
                                                alt_fallback=True,
                                                fallback_extract_func=downloader.extract_zip,
                                                fallback_extract_func_args=(
                                                    str(soundboard_plugin_sounds_dir / os.path.basename(soundpack_urls[0])),
                                                    str(soundboard_plugin_sounds_dir.resolve()),
                                                ),
                                                title="Downloading Soundpack " + soundpack_name)
                    self.set_plugin_setting("sounds_folder", {"type": "folder_open", "accept": "", "value": str(soundboard_plugin_sounds_dir.resolve())})
                    # write text file
                    with open(str(Path(soundboard_plugin_dir / "soundboard_folder.txt").resolve()), "w") as f:
                        f.write(str(soundboard_plugin_sounds_dir.resolve()))
                    self.init()
                    websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Soundpack downloaded and sounds_folder set to default.\nPlease reopen the Plugin Settings to see the new sounds."}))
                if message["value"].startswith("sound_play_btn_"):
                    sound = self.get_plugin_setting(message["value"])
                    if sound != "":
                        wav_numpy = audio_tools.load_wav_to_bytes(sound, target_sample_rate=self.target_sample_rate)
                        # Convert numpy array back to WAV bytes
                        with io.BytesIO() as byte_io:
                            soundfile.write(byte_io, wav_numpy, samplerate=self.target_sample_rate,
                                            format='WAV')  # Explicitly specify format
                            wav_bytes = byte_io.getvalue()

                        self.play_audio_on_device(wav_bytes, settings.GetOption("device_out_index"),
                                                  source_sample_rate=self.target_sample_rate,
                                                  audio_device_channel_num=2,
                                                  target_channels=2,
                                                  input_channels=1,
                                                  dtype="int16")
        else:
            websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Plugin is disabled."}))

    def convert_mp3_to_wav(self, mp3_path, wav_path):
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")

    def stt(self, text, result_obj):
        pass

    def timer(self):
        pass

    def tts(self, text, device_index, websocket_connection=None, download=False):
        pass
