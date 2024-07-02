# ============================================================
# Load and use a second profile settings file for processing audio using Whispering Tiger
# Version 1.0.1
#
# See https://github.com/Sharrnah/whispering
# ============================================================
import json
import time

import Plugins

import VRC_OSCLib
import audio_tools
import audioprocessor
import settings
import pyaudio
import pyaudiowpatch
import threading
import audio_processing_recording
import websocket
from Models.STS import VAD, DeepFilterNet, Noisereduce

from whisper import audio as whisper_audio

from Models.TextTranslation import texttranslate

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = whisper_audio.SAMPLE_RATE
CHUNK = int(SAMPLE_RATE / 10)


class SecondaryProfilePlugin(Plugins.Base):
    settings = None
    py_audio = None
    stream = None

    websocket_server = None

    audio_thread = None
    processor = None
    plugins = []

    active = False

    paused_stt = False
    pause_lock = threading.Lock()
    pause_event = threading.Event()
    pause_timer = None
    pause_end_time = None

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "recording_device_index": {"type": "select_audio", "device_type": "input", "device_api": "all", "value": str(settings.GetOption("device_index"))},
                "settings_file": {"type": "file_open", "accept": ".yaml,.yml", "value": "Profiles/"},
                "vad_enabled": True,
                "btn_start": {"label": "start", "type": "button", "style": "primary"},
                "btn_stop": {"label": "stop", "type": "button", "style": "default"},
                #"btn_update": {"label": "Update Settings", "type": "button", "style": "default"},
                "pause_time_on_main_activity": {"type": "slider", "min": 0.5, "max": 20.0, "step": 0.5, "value": 2.0},

                # processing settings
                "whisper_task": {"type": "select", "value": "transcribe", "values": ["transcribe", "translate"]},
                "stt_enabled": True,
                "osc_sending": False,
                "realtime_frequency_time": {"type": "slider", "min": 0.1, "max": 20, "step": 0.1, "value": 1.0},

                # websocket settings
                "websocket_ip": {"type": "textfield", "value": "0"},
                "websocket_port": {"type": "textfield", "value": "5001"},

                # OSC settings
                "osc_auto_processing_enabled": False,
            },
            settings_groups={
                "General": ["recording_device_index", "settings_file", "vad_enabled", "btn_start", "btn_stop", "pause_time_on_main_activity"],
                #"Settings": ["btn_update", "whisper_task", "stt_enabled", "osc_sending", "realtime_frequency_time"],
                "Settings": ["whisper_task", "stt_enabled", "osc_sending", "realtime_frequency_time"],
                "Websocket": ["websocket_ip", "websocket_port"],
                "OSC": ["osc_auto_processing_enabled"],
            }
        )

    def main_app_before_callback_func_for_pause(self, main_app_obj=None):
        self.set_pause(self.get_plugin_setting("pause_time_on_main_activity"))

    def btn_start(self):
        self.settings = settings.SettingsManager(immutable=True)
        self.settings.load_yaml(self.get_plugin_setting("settings_file"))

        self.settings.SetOption("websocket_ip", "0")
        #self.settings.SetOption("realtime", False)
        self.settings.SetOption("tts_answer", False)
        self.settings.SetOption("audio_processor_caller", "secondary_profile_plugin")
        #self.settings.SetOption("skip_plugins", True)
        self.update_settings()

        self.late_init()

        websocket_ip = self.get_plugin_setting("websocket_ip")
        websocket_port = self.get_plugin_setting("websocket_port")
        if websocket_ip != "0" and websocket_ip != "":
            self.websocket_server = websocket.WebSocketServer(websocket_ip, int(websocket_port), None, None, None,
                                                              debug=False)

        # register callbacks to main app audio callback
        audio_processing_recording.MAIN_APP_BEFORE_CALLBACK_FUNC_LISTS['before_recording_starts_callback_func'].append(self.main_app_before_callback_func_for_pause)
        audio_processing_recording.MAIN_APP_BEFORE_CALLBACK_FUNC_LISTS['before_recording_running_callback_func'].append(self.main_app_before_callback_func_for_pause)
        audio_processing_recording.MAIN_APP_BEFORE_CALLBACK_FUNC_LISTS['before_recording_send_to_queue_callback_func'].append(self.main_app_before_callback_func_for_pause)


    def update_settings(self):
        self.settings.SetOption("whisper_task", self.get_plugin_setting("whisper_task"))
        if not self.paused_stt:
            # skip updating stt_enabled if it is paused
            self.settings.SetOption("stt_enabled", self.get_plugin_setting("stt_enabled"))
        self.settings.SetOption("osc_auto_processing_enabled", self.get_plugin_setting("osc_sending"))
        self.settings.SetOption("whisper_task", self.get_plugin_setting("whisper_task"))
        self.settings.SetOption("realtime_frequency_time", self.get_plugin_setting("realtime_frequency_time"))

    def update_settings_in_callback(self, callback_obj=None):
        self.update_settings()

    def set_pause(self, duration):
        with self.pause_lock:
            if self.pause_timer is not None:
                self.pause_timer.cancel()

            self.pause_event.clear()

            # Calculate the new pause end time
            new_pause_end_time = time.time() + duration
            if self.pause_end_time is None or new_pause_end_time > self.pause_end_time:
                self.pause_end_time = new_pause_end_time

            self.pause_timer = threading.Timer(duration, self._pause_resume)
            self.pause_timer.start()

            if self.settings is not None:
                self.paused_stt = True
                self.settings.SetOption("stt_enabled", False)

    def _pause_resume(self):
        with self.pause_lock:
            current_time = time.time()
            if current_time >= self.pause_end_time:
                self.pause_event.set()
                if self.settings is not None:
                    self.paused_stt = False
                    self.settings.SetOption("stt_enabled", self.get_plugin_setting("stt_enabled"))
                self.pause_end_time = None
            else:
                # Set another timer to resume at the correct time
                remaining_time = self.pause_end_time - current_time
                self.pause_timer = threading.Timer(remaining_time, self._pause_resume)
                self.pause_timer.start()

    def on_event_received(self, message, websocket_connection=None):
        if self.is_enabled(False):
            if "type" not in message:
                return
            if message["type"] == "plugin_button_press":
                if message["value"] == "btn_start":
                    self.btn_start()
                    self.active = True
                if message["value"] == "btn_stop":
                    self.stop_thread()
                    self.active = False
                if message["value"] == "btn_update":
                    self.update_settings()


    def stop_thread(self):
        if self.audio_thread is not None:
            self.audio_thread.join()
            self.audio_thread = None

        if self.processor is not None:
            self.processor = None

        if self.py_audio is not None:
            self.stream.close()
            self.py_audio.terminate()
            self.py_audio = None
            self.stream = None
        print("stopped second processing plugin audio")

    def late_init(self):
        vad_enabled = self.get_plugin_setting("vad_enabled")
        vad_thread_num = int(float(settings.SETTINGS.GetOption("vad_thread_num")))

        osc_ip = settings.SETTINGS.GetOption("osc_ip")
        osc_port = settings.SETTINGS.GetOption("osc_port")
        osc_address = settings.SETTINGS.GetOption("osc_address")

        # load audio filter model
        audio_enhancer = None
        if settings.SETTINGS.GetOption("denoise_audio") == "deepfilter":
            post_filter = settings.SETTINGS.GetOption("denoise_audio_post_filter")
            audio_enhancer = DeepFilterNet.DeepFilterNet(post_filter=post_filter)
        elif settings.SETTINGS.GetOption("denoise_audio") == "noise_reduce":
            audio_enhancer = Noisereduce.Noisereduce()

        if vad_enabled:
            vad_model = VAD.VAD(vad_thread_num)

            # num_samples = 1536
            vad_frames_per_buffer = int(settings.GetOption("vad_frames_per_buffer"))
            vad_model.set_vad_frames_per_buffer(vad_frames_per_buffer)

            default_sample_rate = SAMPLE_RATE

            start_rec_on_volume_threshold = False

            push_to_talk_key = settings.SETTINGS.GetOption("push_to_talk_key")
            if push_to_talk_key == "":
                push_to_talk_key = None
            keyboard_rec_force_stop = False

            # load plugins into array
            self.plugins = []
            # add itself to the list
            self.plugins.append(self)
            plugin_enabled_list = self.settings.GetOption("plugins")
            for plugin in self.base_plugins_list:
                if plugin.__name__ in plugin_enabled_list and plugin_enabled_list[plugin.__name__]:
                    self.plugins.append(plugin(init_settings=self.settings))

            self.processor = audio_processing_recording.AudioProcessor(
                default_sample_rate=default_sample_rate,
                start_rec_on_volume_threshold=start_rec_on_volume_threshold,
                push_to_talk_key=push_to_talk_key,
                keyboard_rec_force_stop=keyboard_rec_force_stop,
                vad_model=vad_model,
                plugins=self.plugins,
                audio_enhancer=audio_enhancer,
                osc_ip=osc_ip,
                osc_port=osc_port,
                chunk=vad_frames_per_buffer,
                channels=CHANNELS,
                sample_format=FORMAT,
                audio_queue=audioprocessor.q,
                settings=self.settings,
                typing_indicator_function=None,
                verbose=False,
                before_callback_called_func=self.update_settings_in_callback
            )
            self.start_audio_stream()


    def audio_thread_run(self):
        # wait a bit before initialization
        time.sleep(8)

        self.py_audio = pyaudiowpatch.PyAudio()

        # call init methods
        for plugin in self.plugins:
            if plugin != self: # skip self
                try:
                    plugin.init()
                    if plugin.is_enabled(False):
                        print(plugin.__class__.__name__ + " is enabled")
                    else:
                        print(plugin.__class__.__name__ + " is disabled")
                except Exception as e:
                    print(f"Error initializing plugin {plugin.__class__.__name__}: {e}")

        # num_samples = 1536
        vad_frames_per_buffer = int(settings.GetOption("vad_frames_per_buffer"))

        plugin_device_index = self.get_plugin_setting("recording_device_index")
        print("plugin_device_index", plugin_device_index)
        device_index = int(self.settings.set_option("device_index", plugin_device_index))
        print("device_index", device_index)
        device_default_in_index = int(settings.GetOption("device_default_in_index"))

        # set default devices if not set
        if device_index is None or device_index < 0:
            device_index = device_default_in_index

        print("Selected Secondary Plugin Device:", device_index)

        # initialize audio stream
        self.stream, needs_sample_rate_conversion, recorded_sample_rate, is_mono = audio_tools.start_recording_audio_stream(
            device_index,
            sample_format=FORMAT,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            chunk=vad_frames_per_buffer,
            py_audio=self.py_audio,
            audio_processor=self.processor,
        )

        # Start the stream
        self.stream.start_stream()

        while self.stream.is_active():
            time.sleep(0.1)

    def start_audio_stream(self):
        self.audio_thread = threading.Thread(target=self.audio_thread_run)
        self.audio_thread.start()

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        self.init()
        pass

    # def stt(self, text, result_obj):
    #     if self.websocket_server is not None:
    #         self.websocket_server.broadcast_message(json.dumps({"type": "processing_data", "data": "debug:: "+text}))

    #def sts(self, wavefiledata, sample_rate):
    #    self.set_pause(2)

    def custom_stt(self, text, result_obj):
        if self.is_enabled(False) and text.strip() != "" and self.active:
            self.send_message(text, result_obj, True)

            for plugin in self.plugins:
                if hasattr(plugin, 'stt'):
                    try:
                        plugin.stt(text, result_obj)
                    except Exception as e:
                        print(f"Error while processing plugin stt in Plugin {plugin.__class__.__name__} over {self.__class__.__name__}: " + str(e))
        return

    def custom_stt_intermediate(self, text, result_obj):
        if self.is_enabled(False) and text.strip() != "" and self.active:
            self.send_message(text, result_obj, False)

            for plugin in self.plugins:
                if hasattr(plugin, 'stt_intermediate'):
                    try:
                        plugin.stt_intermediate(text, result_obj)
                    except Exception as e:
                        print(f"Error while processing plugin stt_intermediate in Plugin {plugin.__class__.__name__} over {self.__class__.__name__}: " + str(e))

        return

    def on_audio_processor_stt_secondary_profile_plugin_call(self, data_obj):
        if self.is_enabled(False) and self.active:
            text = data_obj["text"]
            result_obj = data_obj["result_obj"]
            final_audio = data_obj["final_audio"]

            if final_audio:
                self.custom_stt(text, result_obj)
            else:
                self.custom_stt_intermediate(text, result_obj)


    def send_message(self, predicted_text, result_obj, final_audio):
        osc_ip = self.settings.GetOption("osc_ip")
        osc_address = self.settings.GetOption("osc_address")
        osc_port = self.settings.GetOption("osc_port")
        websocket_ip = self.get_plugin_setting("websocket_ip")

        verbose = self.settings.GetOption("verbose")
        do_txt_translate = self.settings.GetOption("txt_translate")

        predicted_text = predicted_text.strip()
        result_obj["type"] = "transcript"

        # Try to prevent sentence repetition
        sentence_split_language = "english"
        if "language" in result_obj:
            sentence_split_language = result_obj["language"]
        predicted_text = audioprocessor.remove_repetitions(predicted_text, language=sentence_split_language, settings=self.settings)
        if "text" in result_obj:
            result_obj["text"] = predicted_text

        if not predicted_text.lower() in audioprocessor.ignore_list:
            # translate using text translator if enabled
            # translate text realtime or after audio is finished
            if do_txt_translate and self.settings.GetOption("txt_translate_realtime") or \
                    do_txt_translate and not self.settings.GetOption("txt_translate_realtime") and final_audio:
                from_lang = self.settings.GetOption("src_lang")
                to_lang = self.settings.GetOption("trg_lang")
                to_romaji = self.settings.GetOption("txt_romaji")
                predicted_text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(predicted_text, from_lang,
                                                                                             to_lang, to_romaji)
                result_obj["txt_translation"] = predicted_text
                result_obj["txt_translation_source"] = txt_from_lang
                result_obj["txt_translation_target"] = to_lang

            # send realtime processing data to websocket
            if not final_audio and predicted_text.strip() != "":
                self.websocket_server.broadcast_message(json.dumps({"type": "processing_data", "data": predicted_text}))


        # WORKAROUND: prevent it from outputting the initial prompt.
        if predicted_text == self.settings.GetOption("initial_prompt"):
            return

        # Send over OSC
        if osc_ip != "0" and self.get_plugin_setting("osc_auto_processing_enabled") and predicted_text != "":
            osc_notify = final_audio and self.settings.GetOption("osc_typing_indicator")

            osc_send_type = self.settings.GetOption("osc_send_type")
            osc_chat_limit = self.settings.GetOption("osc_chat_limit")
            osc_time_limit = self.settings.GetOption("osc_time_limit")
            osc_scroll_time_limit = self.settings.GetOption("osc_scroll_time_limit")
            osc_initial_time_limit = self.settings.GetOption("osc_initial_time_limit")
            osc_scroll_size = self.settings.GetOption("osc_scroll_size")
            osc_max_scroll_size = self.settings.GetOption("osc_max_scroll_size")
            osc_type_transfer_split = self.settings.GetOption("osc_type_transfer_split")
            osc_type_transfer_split = audioprocessor.replace_osc_placeholders(osc_type_transfer_split, result_obj, self.settings)

            osc_text = predicted_text
            if self.settings.GetOption("osc_type_transfer") == "source":
                osc_text = result_obj["text"]
            elif self.settings.GetOption("osc_type_transfer") == "both":
                osc_text = result_obj["text"] + osc_type_transfer_split + predicted_text
            elif self.settings.GetOption("osc_type_transfer") == "both_inverted":
                osc_text = predicted_text + osc_type_transfer_split + result_obj["text"]

            message = audioprocessor.build_whisper_translation_osc_prefix(result_obj, self.settings) + osc_text

            # delay sending message if it is the final audio and until TTS starts playing
            if final_audio and self.settings.GetOption("osc_delay_until_audio_playback"):
                # wait until is_audio_playing is True or timeout is reached
                delay_timeout = time.time() + self.settings.GetOption("osc_delay_timeout")
                tag = self.settings.GetOption("osc_delay_until_audio_playback_tag")
                tts_answer = self.settings.GetOption("tts_answer")
                if tag == "tts" and tts_answer:
                    while not audio_tools.is_audio_playing(tag=tag) and time.time() < delay_timeout:
                        time.sleep(0.05)

            if osc_send_type == "full":
                VRC_OSCLib.Chat(message, True, osc_notify, osc_address,
                                IP=osc_ip, PORT=osc_port,
                                convert_ascii=self.settings.GetOption("osc_convert_ascii"))
            elif osc_send_type == "chunks":
                VRC_OSCLib.Chat_chunks(message,
                                       nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                       chunk_size=osc_chat_limit, delay=osc_time_limit,
                                       initial_delay=osc_initial_time_limit,
                                       convert_ascii=self.settings.GetOption("osc_convert_ascii"))
            elif osc_send_type == "scroll":
                VRC_OSCLib.Chat_scrolling_chunks(message,
                                                 nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                                 chunk_size=osc_max_scroll_size, delay=osc_scroll_time_limit,
                                                 initial_delay=osc_initial_time_limit,
                                                 scroll_size=osc_scroll_size,
                                                 convert_ascii=self.settings.GetOption("osc_convert_ascii"))
            elif osc_send_type == "full_or_scroll":
                # send full if message fits in osc_chat_limit, otherwise send scrolling chunks
                if len(message.encode('utf-16le')) <= osc_chat_limit * 2:
                    VRC_OSCLib.Chat(message, True, osc_notify, osc_address,
                                    IP=osc_ip, PORT=osc_port,
                                    convert_ascii=self.settings.GetOption("osc_convert_ascii"))
                else:
                    VRC_OSCLib.Chat_scrolling_chunks(message,
                                                     nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                                     chunk_size=osc_chat_limit, delay=osc_scroll_time_limit,
                                                     initial_delay=osc_initial_time_limit,
                                                     scroll_size=osc_scroll_size,
                                                     convert_ascii=self.settings.GetOption("osc_convert_ascii"))

            self.settings.SetOption("plugin_timer_stopped", True)

        # Send to Websocket
        if self.settings.GetOption("websocket_final_messages") and websocket_ip != "0" and final_audio:
            self.websocket_server.broadcast_message(json.dumps(result_obj))
