# ============================================================
# prediction of text emotion plugin for Whispering Tiger
# Version 1.0.4
# See https://github.com/Sharrnah/whispering
# ============================================================
import json
import threading

import Plugins

from transformers import pipeline
import torch

import VRC_OSCLib
import settings
import websocket

COMMANDS = {
    "default": {"param": "WhiteTiger", "value": 0},
    "anger": {"param": "WhiteTiger", "value": 1},
    "sadness": {"param": "WhiteTiger", "value": 1},
    "fear": {"param": "WhiteTiger", "value": 1},
    "joy": {"param": "WhiteTiger", "value": 0},
    "love": {"param": "WhiteTiger", "value": 0},
    "surprise": {"param": "WhiteTiger", "value": 0},
}


class TextEmotionVrcPlugin(Plugins.Base):
    model = None
    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
    bit_length = 32  # can be 32 = 32 float, 16 = 16 float or 8 = 8 int
    device = "CPU"  # can be "CUDA" or "CPU"
    neutral_timer = None
    neutral_time = 15  # 15 seconds by default

    def schedule_neutral_timer(self):
        if self.is_enabled(False) and self.model is not None:
            self.neutral_time = self.get_plugin_setting("neutral_time", self.neutral_time)
            mappings = self.get_plugin_setting("mappings", COMMANDS)

            if self.neutral_timer:
                self.neutral_timer.cancel()

            if "default" in mappings:
                command = mappings["default"]
                self.neutral_timer = threading.Timer(self.neutral_time, self.send_osc_command,
                                                     args=(command['param'], command['value'])
                                                     )
                self.neutral_timer.start()

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "debug": False,
                "device": {"type": "select", "value": "CPU", "values": ["CPU", "CUDA"]},
                "bit_length": 32,
                "mappings": COMMANDS,
                "model_name": "bhadresh-savani/distilbert-base-uncased-emotion",
                "neutral_time": 15,
                "translate": False,
            }
        )

        self.get_plugin_setting("debug")
        self.device = self.get_plugin_setting("device")  # can be "CUDA" or "CPU"
        self.bit_length = self.get_plugin_setting("bit_length")  # can be 32 = 32 float, 16 = 16 float or 8 = 8 int
        self.get_plugin_setting("mappings")
        self.model_name = self.get_plugin_setting("model_name")

        self.neutral_time = self.get_plugin_setting("neutral_time")

        self.get_plugin_setting("translate")

        if self.is_enabled(False):
            if self.model is None:
                try:
                    print(
                        f"{self.model_name} is Loading to {('GPU' if self.device == 'auto' else 'CPU')} using {self.bit_length} bit {('INT' if self.bit_length == 8 else 'float')} precision...")

                    precision = torch.float32
                    match self.bit_length:
                        case 16:  # 16 bit float
                            precision = torch.float16

                        case 8:  # 8 bit int
                            precision = torch.int8

                    self.model = pipeline("text-classification", model=self.model_name, top_k=None,
                                          device=self.device.lower(),
                                          torch_dtype=precision)
                except Exception as e:
                    websocket.BroadcastMessage(json.dumps({"type": "error", "data": str(e)}))

            # Schedule neutral timer
            if self.neutral_time > 0:
                self.schedule_neutral_timer()
        pass

    def predict(self, input_text):
        prediction = self.model(input_text)

        # Sort predictions based on score in descending order
        sorted_predictions = sorted(prediction[0], key=lambda x: x['score'], reverse=True)

        if self.get_plugin_setting("debug", False):
            print(f"Predictions: {sorted_predictions}")

        return sorted_predictions

    def send_osc_command(self, command_parameter, command_value):
        osc_ip = settings.GetOption("osc_ip")
        osc_port = settings.GetOption("osc_port")

        if isinstance(command_value, float):
            VRC_OSCLib.AV3_SetFloat(command_value, command_parameter, osc_ip, osc_port)
        elif isinstance(command_value, int):
            VRC_OSCLib.AV3_SetInt(command_value, command_parameter, osc_ip, osc_port)
        elif isinstance(command_value, bool):
            VRC_OSCLib.AV3_SetBool(command_value, command_parameter, osc_ip, osc_port)

        print(f"Command parameter {command_parameter} value {command_value} sent.")

    def timer(self):
        pass

    def stt(self, text, result_obj):
        if self.is_enabled(False) and self.model is not None:
            mappings = self.get_plugin_setting("mappings", COMMANDS)
            prediction = self.predict(text)

            # Check if the top predicted emotion matches with a command
            if prediction[0]['label'] in mappings:
                command = mappings[prediction[0]['label']]
                self.send_osc_command(command['param'], command['value'])
                self.schedule_neutral_timer()  # reset the neutral timer
        pass

    def tts(self, text, device_index, websocket_connection=None, download=False):
        pass

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        pass
