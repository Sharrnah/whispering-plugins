# ============================================================
# RVC Speech to Speech Plugin for Whispering Tiger
# V1.1.5
# RVC WebUI: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
# Whispering Tiger: https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import base64
import io
import json
import queue
import random
import shutil
import threading
import time
import wave

import librosa
import numpy as np
import torch

import Plugins
from pathlib import Path
import os
import sys

from scipy.io.wavfile import write as write_wav
from Models.STS import DeepFilterNet

import audio_tools
import downloader
import settings
import websocket

rvc_sts_plugin_dir = Path(Path.cwd() / "Plugins" / "rvc_sts_plugin")
os.makedirs(rvc_sts_plugin_dir, exist_ok=True)
rvc_sts_plugin_weights_dir = Path(Path.cwd() / "Plugins" / "rvc_sts_plugin" / "weights")
os.makedirs(rvc_sts_plugin_weights_dir, exist_ok=True)

# for realtime
import sounddevice as sd
import torch.nn.functional as F

rvc_webui_dependency = {
    "urls": [
        "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/archive/f431f8fb3f13aa6dfedf33383f70de35fe07dfbd.zip",
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:projects/rvc-plugin/Retrieval-based-Voice-Conversion-WebUI-f431f8fb3f13aa6dfedf33383f70de35fe07dfbd.zip",
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:projects/rvc-plugin2/Retrieval-based-Voice-Conversion-WebUI-f431f8fb3f13aa6dfedf33383f70de35fe07dfbd.zip"
     ],
    "sha256": "75eb5f3bcadf9bb56ef73415d56940ded6d3c2d1feae34b5252ae15266459d73",
    "zip_path": "Retrieval-based-Voice-Conversion-WebUI-f431f8fb3f13aa6dfedf33383f70de35fe07dfbd",
    "target_path": "Retrieval-based-Voice-Conversion-WebUI"
}
rvc_models = {
    "urls": [
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:projects/rvc-plugin/rvc_models.zip",
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:projects/rvc-plugin2/rvc_models.zip"
    ],
    "sha256": "75df758e11605fde28f3d82cf7415503deee0fb4de95d838db8b320474823816"
}
rmvpe_model = {
    "urls": [
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:projects/rvc-plugin/rmvpe_model.zip",
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:projects/rvc-plugin2/rmvpe_model.zip"
    ],
    "sha256": "63d9f0b001eb0749a0ec6a7f12d7b5193b1b54a1a259fcfc4201eb81d7dc0627"
}

rvc_infer_script = {
    "urls": [
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:projects/rvc-plugin/rvc_infer.py",
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:projects/rvc-plugin2/rvc_infer.py"
    ],
    "sha256": "652b85bdfb9d6190cf75443f00064b9f5039fce975fac0ccdbaae5fde8f8df46"
}
rvc_realtime_infer_script = {
    "urls": [
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:projects/rvc-plugin/rvc_for_realtime_patched.py",
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:projects/rvc-plugin2/rvc_for_realtime_patched.py"
    ],
    "sha256": "c3f1bc3796f2e67b3a15aad76ba804f51da1416fbc7c6544bc01e3b3576b6ed5"
}

CONSTANTS = {
    "DISABLED": 'Disabled',
    "STS": 'Own Voice',
    "STS_RT": 'Own Voice (Realtime)',
    "SILERO_TTS": 'Integrated Text-to-Speech (Silero TTS)',
    "PLUGIN_TTS": 'Plugin Text-to-Speech',
}

#sys.path.append(str(rvc_sts_plugin_dir.resolve()))
#sys.path.append(os.path.join(rvc_sts_plugin_dir, "Retrieval-based-Voice-Conversion-WebUI"))
#
#from tools import rvc_for_realtime
#from configs.config import Config
#from tools.torchgate import TorchGate
#
#import torchaudio.transforms as tat
#import threading
#import time
#import sounddevice as sd


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


class RVCStsPlugin(Plugins.Base):
    output_sample_rate = 16000
    audio_denoiser = None

    debug = False
    stop_vr_thread = False
    flag_vc = False

#    audio_processing_queue = queue.Queue()
#
#    device_index = None
#    audio_playback_queue = None
#    streaming_playback_thread = None
#
#    audio_buffer = b''
#    audio_buffer_duration = 0  # Duration of audio in buffer in milliseconds
#    target_duration = 500  # Target duration in milliseconds for processing

    def model_file_valid(self, file_path: str):
        # check if file exists
        if os.path.exists(file_path):
            return True
        return False

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        pass

    def init(self):
        self.init_plugin_settings(
            {
                # voice conversion settings
                "index_rate": {"type": "slider", "min": 0, "max": 1, "step": 0.01, "value": 0.75},
                "f0up_key": {"type": "slider", "min": -50.0, "max": 50.0, "step": 0.1, "value": -13.0},
                "f0up_key_info": {"label": "f0up_key (pitch setting) : lower (ca.-12) if voice conversion is female to male, higher (ca.+12) if male to female.", "type": "label", "style": "left"},
                "f0method": {"type": "select", "value": "harvest", "values": ["harvest", "pm", "crepe", "rmvpe"]},
                "filter_radius": {"type": "slider", "min": 1, "max": 10, "step": 1, "value": 3},
                "rms_mix_rate": {"type": "slider", "min": 0, "max": 1, "step": 0.01, "value": 0.25},
                "protect": {"type": "slider", "min": 0, "max": 1, "step": 0.01, "value": 0.33},

                # audio conversion
                "audio_file": {"type": "file_open", "accept": ".wav", "value": ""},
                "convert_btn": {"label": "convert audio file", "type": "button", "style": "primary"},
                "voice_change_source": {"type": "select", "value": CONSTANTS["STS"],
                                        "values": [value for key, value in CONSTANTS.items()]},

                # realtime settings
                "rt_input_device_index": {"type": "select_audio", "device_api": "mme", "device_type": "input", "value": str(settings.GetOption("device_index"))},
                "rt_input_noise_reduce": False,
                "rt_output_device_index": {"type": "select_audio", "device_api": "mme", "device_type": "output", "value": str(settings.GetOption("device_out_index"))},
                "rt_output_noise_reduce": False,
                "rt_threshold": {"type": "slider", "min": -60.0, "max": 0.0, "step": 1, "value": -60.0},
                "rt_block_time": {"type": "slider", "min": 0.1, "max": 5.0, "step": 0.1, "value": 1.0},
                "rt_extra_time": {"type": "slider", "min": 0.1, "max": 10.0, "step": 0.1, "value": 2.0},
                "rt_crossfade_time": {"type": "slider", "min": 0.01, "max": 5.0, "step": 0.01, "value": 0.04},
                "rt_restart_btn": {"label": "start / restart / stop Realtime", "type": "button", "style": "primary"},
                "rt_restart_btn_info": {"label": "To start, set 'voice_change_source' to 'Own Voice (Realtime)'.\nThen Press the Button.\nFor settings to take effect, press it again.", "type": "label", "style": "left"},


                # model settings
                "model_file": {"type": "file_open", "accept": ".pth", "value": ""},
                "index_file": {"type": "file_open", "accept": ".index", "value": ""},
                "model_load_btn": {"label": "Load model", "type": "button", "style": "primary"},
                "half_precision": False,
                "device": {"type": "select", "value": "cpu:0",
                           "values": ["cpu:0", "cpu:1", "cpu:2", "cuda:0", "cuda:1", "cuda:2"]},
                "result_noise_filter": False,
                "unload_on_finish": False,
                "debug": False,
            },
            settings_groups={
                "General": ["index_rate", "f0up_key", "f0up_key_info", "f0method", "filter_radius", "rms_mix_rate", "protect"],
                "Audio conversion": ["voice_change_source", "audio_file", "convert_btn"],
                "Realtime": ["rt_input_device_index", "rt_input_noise_reduce", "rt_output_device_index",
                             "rt_output_noise_reduce", "rt_threshold", "rt_restart_btn", "rt_restart_btn_info",
                             "rt_block_time", "rt_extra_time", "rt_crossfade_time"],
                "Model": ["model_file", "index_file", "model_load_btn", "half_precision", "device", "result_noise_filter", "unload_on_finish", "debug"],
            }
        )

        if self.is_enabled(False) and self.model_file_valid(self.get_plugin_setting("model_file")):
            # download infer script
            if not Path(rvc_sts_plugin_dir / "rvc_infer.py").is_file() or downloader.sha256_checksum(str(Path(rvc_sts_plugin_dir / "rvc_infer.py").resolve())) != rvc_infer_script["sha256"]:
                # delete rvc_infer.py if it already exists
                if Path(rvc_sts_plugin_dir / "rvc_infer.py").is_file():
                    os.remove(str(Path(rvc_sts_plugin_dir / "rvc_infer.py").resolve()))

                infer_script_url = random.choice(rvc_infer_script["urls"])
                downloader.download_extract([infer_script_url],
                                            str(rvc_sts_plugin_dir.resolve()),
                                            rvc_infer_script["sha256"],
                                            alt_fallback=True,
                                            title="RVC Inference script", extract_format="none")

            # download rvc_webui
            if not Path(rvc_sts_plugin_dir / rvc_webui_dependency["target_path"] / "infer-web.py").is_file():
                print("rvc_webui downloading...")
                # download from random url in list
                voice_clone_url = random.choice(rvc_webui_dependency["urls"])
                downloader.download_extract([voice_clone_url],
                                            str(rvc_sts_plugin_dir.resolve()),
                                            rvc_webui_dependency["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(rvc_sts_plugin_dir / os.path.basename(voice_clone_url)),
                                                str(rvc_sts_plugin_dir.resolve()),
                                            ),
                                            title="RVC", extract_format="zip")
                # rename folder
                shutil.move(str(rvc_sts_plugin_dir / rvc_webui_dependency["zip_path"]), str(rvc_sts_plugin_dir / rvc_webui_dependency["target_path"]))

            # download realtime infer script
            if not Path(rvc_sts_plugin_dir / "Retrieval-based-Voice-Conversion-WebUI" / "tools" / "rvc_for_realtime_patched.py").is_file() or downloader.sha256_checksum(str(Path(rvc_sts_plugin_dir / "Retrieval-based-Voice-Conversion-WebUI" / "tools" / "rvc_for_realtime_patched.py").resolve())) != rvc_realtime_infer_script["sha256"]:
                # delete rvc_infer.py if it already exists
                if Path(rvc_sts_plugin_dir / "Retrieval-based-Voice-Conversion-WebUI" / "tools" / "rvc_for_realtime_patched.py").is_file():
                    os.remove(str(Path(rvc_sts_plugin_dir / "Retrieval-based-Voice-Conversion-WebUI" / "tools" / "rvc_for_realtime_patched.py").resolve()))

                realtime_infer_script_url = random.choice(rvc_realtime_infer_script["urls"])
                realtime_infer_script_target_download_dir = Path(rvc_sts_plugin_dir / "Retrieval-based-Voice-Conversion-WebUI" / "tools")
                downloader.download_extract([realtime_infer_script_url],
                                            str(realtime_infer_script_target_download_dir.resolve()),
                                            rvc_realtime_infer_script["sha256"],
                                            alt_fallback=True,
                                            title="RVC Realtime Inference script", extract_format="none")

            rvc_models_path = Path(rvc_sts_plugin_dir / rvc_webui_dependency["target_path"] / "assets")
            if not Path(rvc_models_path / "hubert" / "hubert_base.pt").is_file():
                print("rvc models downloading...")
                # download from random url in list
                rvc_model_url = random.choice(rvc_models["urls"])
                downloader.download_extract([rvc_model_url],
                                            str(rvc_models_path.resolve()),
                                            rvc_models["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(rvc_models_path / os.path.basename(rvc_model_url)),
                                                str(rvc_models_path.resolve()),
                                            ),
                                            title="RVC Models", extract_format="zip")

            if not Path(rvc_models_path / "rmvpe" / "rmvpe.pt").is_file():
                print("rmvpe model downloading...")
                # download from random url in list
                rmvpe_model_url = random.choice(rmvpe_model["urls"])
                downloader.download_extract([rmvpe_model_url],
                                            str(rvc_models_path.resolve()),
                                            rmvpe_model["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(rvc_models_path / os.path.basename(rmvpe_model_url)),
                                                str(rvc_models_path.resolve()),
                                            ),
                                            title="rmvpe model", extract_format="zip")

            sys.path.append(str(rvc_sts_plugin_dir.resolve()))
            sys.path.append(os.path.join(rvc_sts_plugin_dir, "Retrieval-based-Voice-Conversion-WebUI"))

            rvc_path = self.get_plugin_setting("model_file")
            self.index_path = self.get_plugin_setting("index_file")

            # device = "cuda:0"
            device = self.get_plugin_setting("device")
            # is_half = True
            is_half = self.get_plugin_setting("half_precision")

            #
            #self.gui_config = GUIConfig()
            #self.config = Config()
            #
            #self.gui_config.pitch = self.get_plugin_setting("f0up_key")
            #self.gui_config.pth_path = rvc_path
            #self.gui_config.index_path = self.index_path
            #
            #input_devices, output_devices, _, _ = self.get_devices()
            #(
            #    input_devices,
            #    output_devices,
            #    input_device_indices,
            #    output_device_indices,
            #) = self.get_devices()
            #sd.default.device[0] = input_device_indices[
            #    input_devices.index(settings.GetOption("audio_input_device")[:32] + " (MME)")
            #]
            #sd.default.device[1] = output_device_indices[
            #    output_devices.index(settings.GetOption("audio_output_device")[:32] + " (MME)")
            #]
            from rvc_infer import get_vc, vc_single, release_model

            if (self.get_plugin_setting("voice_change_source") != CONSTANTS["STS_RT"] and self.get_plugin_setting("voice_change_source") != CONSTANTS["DISABLED"]) and self.model_file_valid(self.get_plugin_setting("model_file")):
                self.vc_single = vc_single
                self.release_model = release_model
                get_vc(rvc_path, device, is_half)

            #############################
            ## realtime voice conversion
            #############################
            self.n_cpu = min(2, 8)
            self.inp_q = queue.Queue()
            self.opt_q = queue.Queue()
            ##from multiprocessing import Queue, cpu_count
            try:
                import torchaudio.transforms as tat
                from rvc_infer import rvc_for_realtime, TorchGate
                self.rvc_for_realtime = rvc_for_realtime
                self.TorchGate = TorchGate
                self.tat = tat
                if self.get_plugin_setting("voice_change_source") == CONSTANTS["STS_RT"] and self.model_file_valid(self.get_plugin_setting("model_file")):
                    self.start_vc()
            except ImportError as e:
                print("Error initializing realtime dependencies: " + str(e))


        if self.is_enabled(False) and not self.model_file_valid(self.get_plugin_setting("model_file")):
            websocket.BroadcastMessage(json.dumps({"type": "info",
                                                   "data": "No model file found. Please select a model file first."}))
        #if self.is_enabled(False):
        #    if self.get_plugin_setting("voice_change_source") == CONSTANTS["STS_RT"]:
        #        self.audio_playback_queue, self.streaming_playback_thread = audio_tools.start_streaming_audio_playback(
        #            self.device_index,
        #            channels=2,
        #            sample_rate=self.output_sample_rate,
        #        )


    def load_audio_file(self, wav_file_path):
        with wave.open(wav_file_path, 'rb') as wav_file:
            # Extract Raw Audio from Wav File
            signal = wav_file.readframes(-1)

            # Get the number of channels and sample width
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()

            # Determine the appropriate numpy data type for the audio array
            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            float_dtype_map = {3: np.float32}  # 3 is just a placeholder for 32-bit float sample width
            if sample_width in dtype_map:
                dtype = dtype_map[sample_width]
                max_value = np.iinfo(dtype).max
                audio = np.frombuffer(signal, dtype=dtype)
                audio_normalized = (audio / max_value).astype(np.float32)
            elif sample_width == 4 and wav_file.getcomptype() == 'IEEE_FLOAT':
                # For 32-bit float WAV files, data is already in float32 format
                dtype = float_dtype_map[3]
                audio_normalized = np.frombuffer(signal, dtype=dtype)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            # If the WAV file is stereo (2 channels), down-mix to mono
            #is_mono = True
            #if n_channels == 2:
            #    is_mono = False
            #    # This example averages the two channels to down-mix to mono
            #    # Replace the following line with your own downmixing function as needed
            #    #audio = audio.reshape(-1, 2).mean(axis=1)


            # resample the audio
            audio = audio_tools.resample_audio(audio_normalized, frame_rate, self.output_sample_rate, target_channels=1, input_channels=n_channels, dtype="float32")

            return audio

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

    # bytes_dtype is used if audio is bytes
    def do_conversion(self, audio, sample_rate, bytes_dtype="int16"):
        # index_rate = 0.75
        index_rate = self.get_plugin_setting("index_rate")
        # f0up_key = -6
        f0up_key = self.get_plugin_setting("f0up_key")
        # filter_radius = 3
        filter_radius = self.get_plugin_setting("filter_radius")
        # rms_mix_rate = 0.25
        rms_mix_rate = self.get_plugin_setting("rms_mix_rate")
        # protect = 0.33
        protect = self.get_plugin_setting("protect")
        # f0method = "harvest"  # harvest or pm
        f0method = self.get_plugin_setting("f0method")  # harvest or pm

        if isinstance(audio, bytes):
            b_dtype = np.int16
            if bytes_dtype == "float32":
                b_dtype = np.float32
            audio = np.frombuffer(audio, dtype=b_dtype)

        if audio.dtype == np.float32:
            wav_data_float32 = audio
        else:
            wav_data_int16 = np.frombuffer(audio, dtype=np.int16)
            wav_data_float32 = wav_data_int16.astype(np.float32) / np.iinfo(np.int16).max

        try:
            audio_array = self.vc_single(0, wav_data_float32, f0up_key, None, f0method, self.index_path, index_rate,
                                         filter_radius=filter_radius, resample_sr=sample_rate,
                                         rms_mix_rate=rms_mix_rate, protect=protect)
        except Exception as e:
            print("error. falling back: ", e)
            audio_array = self.vc_single(0, wav_data_float32, f0up_key, None, 'pm', self.index_path, index_rate,
                                         filter_radius=filter_radius, resample_sr=sample_rate,
                                         rms_mix_rate=rms_mix_rate, protect=protect)

        if self.get_plugin_setting("result_noise_filter"):
            if self.audio_denoiser is None:
                self.audio_denoiser = DeepFilterNet.DeepFilterNet(post_filter=False)
            if self.audio_denoiser is not None:
                audio_array = self.audio_denoiser.enhance_audio(audio_array)

        if self.get_plugin_setting("unload_on_finish"):
            self.release_model()

        return audio_array

    def sts(self, wavefiledata, sample_rate):
        if self.is_enabled(False) and self.get_plugin_setting("voice_change_source") == CONSTANTS["STS"] and self.model_file_valid(self.get_plugin_setting("model_file")) and settings.GetOption("tts_answer"):
            audio_array = self.do_conversion(wavefiledata, sample_rate, bytes_dtype="int16")

            # create wav audio for playback
            buff = io.BytesIO()
            write_wav(buff, sample_rate, audio_array)
            buff.seek(0)

            device_index = settings.GetOption("device_out_index")
            if device_index is None or device_index == -1:
                device_index = settings.GetOption("device_default_out_index")

            self.play_audio_on_device(buff.getvalue(), device_index,
                                      source_sample_rate=sample_rate,
                                      audio_device_channel_num=2,
                                      target_channels=2,
                                      input_channels=1,
                                      dtype="int16"
                                      )

    @staticmethod
    def calculate_duration(num_bytes, sample_rate):
        # Assuming 16-bit (2 bytes) samples and 2 channels
        sample_duration_ms = 1000 / sample_rate
        return (num_bytes / 2 / 2) * sample_duration_ms

    #    def sts_chunk(self, wavefiledata, sample_rate):
    #        if self.is_enabled(False) and self.streaming_playback_thread is not None and self.get_plugin_setting("voice_change_source") == CONSTANTS["STS_RT"] and self.model_file_valid(self.get_plugin_setting("model_file")) and settings.GetOption("tts_answer"):
    #            # Convert to bytearray before extending
    #            buffer_data = wavefiledata.tobytes() if isinstance(wavefiledata, np.ndarray) else wavefiledata
    #            self.audio_buffer += buffer_data
    #            self.audio_buffer_duration += self.calculate_duration(len(buffer_data), sample_rate)
    #
    #            if self.audio_buffer_duration >= self.target_duration:
    #                audio_array = self.do_conversion(self.audio_buffer, sample_rate, bytes_dtype="int16")
    #                self.audio_playback_queue.put(audio_array)
    #
    #                # Reset buffer
    #                self.audio_buffer = b''
    #                self.audio_buffer_duration = 0



    def get_devices(self, update: bool = True):
        """获取设备列表"""
        if update:
            sd._terminate()
            sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        print("hostAPIs")
        print(hostapis)
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        input_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_input_channels"] > 0
        ]
        output_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_output_channels"] > 0
        ]
        input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0
        ]
        output_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_output_channels"] > 0
        ]
        return (
            input_devices,
            output_devices,
            input_devices_indices,
            output_devices_indices,
        )

    def set_devices(self, input_device, output_device):
        """设置输出设备"""
        # (
        #     input_devices,
        #     output_devices,
        #     input_device_indices,
        #     output_device_indices,
        # ) = self.get_devices()
        #
        # print("input_devices")
        # print(input_devices)
        # print("output_devices")
        # print(output_devices)
        #
        # sd.default.device[0] = input_device_indices[
        #     input_devices.index(input_device)
        # ]
        # sd.default.device[1] = output_device_indices[
        #     output_devices.index(output_device)
        # ]

        sd.default.device[0] = input_device
        sd.default.device[1] = output_device
        printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
        printt("Output device: %s:%s", str(sd.default.device[1]), output_device)

    def start_vc(self):
        print("starting realtime RVC...")
        input_device = int(self.get_plugin_setting("rt_input_device_index", settings.GetOption("device_index")))
        output_device = int(self.get_plugin_setting("rt_output_device_index", settings.GetOption("device_out_index")))

        print("setting audio devices...")
        print("input_device: ", input_device)
        print("output_device: ", output_device)

        self.set_devices(input_device, output_device)

        self.debug = self.get_plugin_setting("debug")
        # index_rate = 0.75
        index_rate = self.get_plugin_setting("index_rate")
        # f0up_key = -6
        f0up_key = self.get_plugin_setting("f0up_key")
        # filter_radius = 3
        filter_radius = self.get_plugin_setting("filter_radius")
        # rms_mix_rate = 0.25
        rms_mix_rate = self.get_plugin_setting("rms_mix_rate")
        # protect = 0.33
        protect = self.get_plugin_setting("protect")
        # f0method = "harvest"  # harvest or pm
        self.f0method = self.get_plugin_setting("f0method")  # harvest or pm
        device = self.get_plugin_setting("device")
        self.device = device

        rvc_path = self.get_plugin_setting("model_file")
        self.index_path = self.get_plugin_setting("index_file")

        is_half = self.get_plugin_setting("half_precision", True)

        I_noise_reduce = self.get_plugin_setting("rt_input_noise_reduce", False)
        O_noise_reduce = self.get_plugin_setting("rt_output_noise_reduce", False)

        # config
        #self.samplerate: int = 40000
        self.block_time: float = self.get_plugin_setting("rt_block_time", 1.0)  # s
        self.crossfade_time: float = self.get_plugin_setting("rt_crossfade_time", 0.04)
        self.extra_time: float = self.get_plugin_setting("rt_extra_time", 2.0)
        #self.threhold: int = -60
        self.threhold: int = int(self.get_plugin_setting("rt_threshold", -60))

        self.function = "vc"

        config_dict = {
            "dml": True,
            "device": device,
            #"use_jit": True,
            "use_jit": False,
            "is_half": is_half,
            "I_noise_reduce": I_noise_reduce,
            "O_noise_reduce": O_noise_reduce,
            "rms_mix_rate": rms_mix_rate,
            "threhold": self.threhold,
        }

        self.config = Config(config_dict)

        torch.cuda.empty_cache()
        self.flag_vc = True
        self.rvc = self.rvc_for_realtime.RVC(
            f0up_key,
            rvc_path,
            self.index_path,
            index_rate,
            self.n_cpu,
            self.inp_q,
            self.opt_q,
            self.config,
            self.rvc if hasattr(self, "rvc") else None,
        )
        self.rvc.set_debug(self.debug)
        self.samplerate = self.rvc.tgt_sr
        self.zc = self.rvc.tgt_sr // 100
        self.block_frame = (
                int(
                    np.round(
                        self.block_time
                        * self.samplerate
                        / self.zc
                    )
                )
                * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
                int(
                    np.round(
                        self.crossfade_time
                        * self.samplerate
                        / self.zc
                    )
                )
                * self.zc
        )
        self.sola_search_frame = self.zc
        self.extra_frame = (
                int(
                    np.round(
                        self.extra_time
                        * self.samplerate
                        / self.zc
                    )
                )
                * self.zc
        )
        self.input_wav: torch.Tensor = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=device,
            dtype=torch.float32,
            )
        self.input_wav_res: torch.Tensor = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=device,
            dtype=torch.float32,
            )
        self.pitch: np.ndarray = np.zeros(
            self.input_wav.shape[0] // self.zc,
            dtype="int32",
            )
        self.pitchf: np.ndarray = np.zeros(
            self.input_wav.shape[0] // self.zc,
            dtype="float64",
            )
        self.sola_buffer: torch.Tensor = torch.zeros(
            self.crossfade_frame, device=device, dtype=torch.float32
        )
        self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
        self.output_buffer: torch.Tensor = self.input_wav.clone()
        self.res_buffer: torch.Tensor = torch.zeros(
            2 * self.zc, device=device, dtype=torch.float32
        )
        self.valid_rate = 1 - (self.extra_frame - 1) / self.input_wav.shape[0]
        self.fade_in_window: torch.Tensor = (
                torch.sin(
                    0.5
                    * np.pi
                    * torch.linspace(
                        0.0,
                        1.0,
                        steps=self.crossfade_frame,
                        device=device,
                        dtype=torch.float32,
                    )
                )
                ** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
        self.resampler = self.tat.Resample(
            orig_freq=self.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(device)
        self.tg = self.TorchGate(
            sr=self.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(device)
        self.thread_vc = threading.Thread(target=self.soundinput)
        self.thread_vc.start()

    def soundinput(self):
        """
        接受音频输入
        """
        #extra_settings = sd.WasapiSettings(auto_convert=True)
        channels = 1 if sys.platform == "darwin" else 2
        with sd.Stream(
                channels=channels,
                callback=self.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.samplerate,
                dtype="float32",
                #extra_settings=extra_settings
        ) as stream:
            global stream_latency
            stream_latency = stream.latency[-1]
            while self.flag_vc:
                time.sleep(self.block_time)
                if self.debug:
                    printt("Audio block passed.")
            if self.stop_vr_thread:
                stream.stop()
                self.stop_vr_thread = False
                if hasattr(self, "rvc") and self.rvc is not None:
                    self.rvc = None
        printt("ENDing VC")

    def audio_callback(
            self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):
        start_time = time.perf_counter()
        #threhold = -55
        self.zc = self.rvc.tgt_sr // 100

        indata = librosa.to_mono(indata.T)
        if self.config.threhold > -60:
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )
            db_threhold = (
                    librosa.amplitude_to_db(rms, ref=1.0)[0] < self.config.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
        self.input_wav[: -self.block_frame] = self.input_wav[
                                              self.block_frame :
                                              ].clone()
        self.input_wav[-self.block_frame :] = torch.from_numpy(indata).to(
            self.device
        )
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                                                      self.block_frame_16k :
                                                      ].clone()

        # input noise reduction and resampling
        if self.config.I_noise_reduce and self.function == "vc":
            input_wav = self.input_wav[
                        -self.crossfade_frame - self.block_frame - 2 * self.zc :
                        ]
            input_wav = self.tg(
                input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
            )[0, 2 * self.zc :]
            input_wav[: self.crossfade_frame] *= self.fade_in_window
            input_wav[: self.crossfade_frame] += (
                    self.nr_buffer * self.fade_out_window
            )
            self.nr_buffer[:] = input_wav[-self.crossfade_frame :]
            input_wav = torch.cat(
                (self.res_buffer[:], input_wav[: self.block_frame])
            )
            self.res_buffer[:] = input_wav[-2 * self.zc :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                input_wav
            )[160:]
        else:
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                self.input_wav[-self.block_frame - 2 * self.zc :]
            )[160:]
        # infer
        if self.function == "vc":
            f0_extractor_frame = self.block_frame_16k + 800
            if self.f0method == "rmvpe":
                f0_extractor_frame = (
                        5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
                )
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.input_wav_res[-f0_extractor_frame:].cpu().numpy(),
                self.block_frame_16k,
                self.valid_rate,
                self.pitch,
                self.pitchf,
                self.f0method,
            )
            infer_wav = infer_wav[
                        -self.crossfade_frame - self.sola_search_frame - self.block_frame :
                        ]
        else:
            infer_wav = self.input_wav[
                        -self.crossfade_frame - self.sola_search_frame - self.block_frame :
                        ].clone()
        # output noise reduction
        if (self.config.O_noise_reduce and self.function == "vc") or (
                self.config.I_noise_reduce and self.function == "im"
        ):
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                                                      self.block_frame :
                                                      ].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        # volume envelop mixing
        if self.config.rms_mix_rate < 1 and self.function == "vc":
            rms1 = librosa.feature.rms(
                y=self.input_wav_res[-160 * infer_wav.shape[0] // self.zc :]
                .cpu()
                .numpy(),
                frame_length=640,
                hop_length=160,
                )
            rms1 = torch.from_numpy(rms1).to(self.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(
                rms1 / rms2, torch.tensor(1 - self.config.rms_mix_rate)
            )
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[
                     None, None, : self.crossfade_frame + self.sola_search_frame
                     ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.crossfade_frame, device=self.device),
                )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        if self.debug:
            printt("sola_offset = %d", int(sola_offset))
        infer_wav = infer_wav[
                    sola_offset : sola_offset + self.block_frame + self.crossfade_frame
                    ]
        infer_wav[: self.crossfade_frame] *= self.fade_in_window
        infer_wav[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[-self.crossfade_frame :]
        if sys.platform == "darwin":
            outdata[:] = (
                infer_wav[: -self.crossfade_frame].cpu().numpy()[:, np.newaxis]
            )
        else:
            outdata[:] = (
                infer_wav[: -self.crossfade_frame].repeat(2, 1).t().cpu().numpy()
            )
        total_time = time.perf_counter() - start_time
        #self.window["infer_time"].update(int(total_time * 1000))
        if self.debug:
            printt("Infer time: %.2f", total_time)

    def on_event_received(self, message, websocket_connection=None):
        if self.is_enabled(False):
            if "type" not in message:
                return
            if message["type"] == "plugin_button_press":
                if message["value"] == "model_load_btn":
                    self.init()
                    websocket.BroadcastMessage(json.dumps({"type": "info",
                                                           "data": "Model loaded."}))
                    pass
                if message["value"] == "convert_btn":
                    input_sample_rate = 16000
                    audio_file = self.get_plugin_setting("audio_file")
                    audio_data = self.load_audio_file(audio_file)
                    audio_array = self.do_conversion(audio_data, input_sample_rate)

                    buff = io.BytesIO()
                    write_wav(buff, input_sample_rate, audio_array)
                    buff.seek(0)

                    wav_data = base64.b64encode(buff.getvalue()).decode('utf-8')
                    websocket.AnswerMessage(websocket_connection, json.dumps({"type": "tts_save", "wav_data": wav_data}))
                    pass
                if message["value"] == "rt_restart_btn":
                    print("pressed rt_restart_btn")
                    if hasattr(self, "thread_vc") and self.thread_vc is not None and self.thread_vc.is_alive():
                        print("stopping realtime thread...")
                        self.flag_vc = False
                        self.stop_vr_thread = True
                        self.thread_vc.join()
                        self.thread_vc = None

                        # wait until self.rvc is None.
                        max_wait_time = 5  # maximum wait time in seconds
                        start_time = time.time()
                        while self.rvc is not None and (time.time() - start_time) < max_wait_time:
                            time.sleep(0.1)
                        if self.rvc is not None:
                            print("Warning: Maximum wait time exceeded. Proceeding with realtime thread stopped.")
                        else:
                            print("realtime thread stopped.")
                    else:
                        if self.is_enabled(False) and self.get_plugin_setting("voice_change_source") != CONSTANTS["STS_RT"] and self.get_plugin_setting("voice_change_source") == CONSTANTS["DISABLED"]:
                            websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Realtime voice change is disabled. Go to Audio Conversion -> voice_change_source\nand set to Realtime."}))

                    if not self.is_enabled(False):
                        websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Plugin is disabled."}))
                        return

                    if self.is_enabled(False) and self.get_plugin_setting("voice_change_source") == CONSTANTS["STS_RT"] and self.model_file_valid(self.get_plugin_setting("model_file")):
                        self.start_vc()
                    pass

    def on_silero_tts_after_audio_call(self, data_obj):
        if self.is_enabled(False) and self.get_plugin_setting("voice_change_source") == CONSTANTS["SILERO_TTS"] and self.model_file_valid(self.get_plugin_setting("model_file")):
            audio = data_obj['audio']
            sample_rate = 48000
            # tensor to numpy
            audio_tmp = audio.detach().cpu().numpy()
            # from float32 to int16
            audio_tmp = audio_tools.convert_audio_datatype_to_integer(audio_tmp)
            # to bytes
            buff = io.BytesIO()
            write_wav(buff, sample_rate, audio_tmp)

            audio_tmp = audio_tools.resample_audio(buff.read(), sample_rate, self.output_sample_rate, target_channels=1, input_channels=1, dtype="int16")
            audio_tmp = self.do_conversion(audio_tmp, sample_rate, bytes_dtype="int16")
            # back to float32
            audio_tmp = audio_tools.convert_audio_datatype_to_float(audio_tmp)
            # back to tensor
            audio = torch.from_numpy(audio_tmp)

            data_obj['audio'] = audio
            return data_obj
        return None

    def on_plugin_tts_after_audio_call(self, data_obj):
        if self.is_enabled(False) and self.get_plugin_setting("voice_change_source") == CONSTANTS["PLUGIN_TTS"] and self.model_file_valid(self.get_plugin_setting("model_file")):
            audio = data_obj['audio']
            sample_rate = data_obj['sample_rate']

            audiodata = audio
            if hasattr(audio, 'getvalue'):
                audiodata = audio.getvalue()

            loaded_audio = audio_tools.resample_audio(audiodata, sample_rate, self.output_sample_rate, target_channels=1, input_channels=1, dtype="int16")
            wav_rvc = self.do_conversion(loaded_audio, sample_rate, bytes_dtype="int16")
            raw_data = audio_tools.numpy_array_to_wav_bytes(wav_rvc, sample_rate)

            if hasattr(audio, 'getvalue'):
                data_obj['audio'] = raw_data
            elif hasattr(raw_data, 'getvalue'):
                data_obj['audio'] = raw_data.getvalue()
            else:
                return None

            return data_obj

        return None


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)
