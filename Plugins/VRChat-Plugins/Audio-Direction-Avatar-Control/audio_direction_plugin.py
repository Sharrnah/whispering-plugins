# ============================================================
# Sends volume and audio direction over OSC using Whispering Tiger
# Version 1.1.1
#
# See https://github.com/Sharrnah/whispering
# Inspired by https://github.com/Codel1417/VRC-OSC-Audio-Reaction
# ============================================================

import Plugins

import VRC_OSCLib
import audio_tools
import settings
import pyaudiowpatch as pyaudio
import numpy as np
import threading


class AudioDirectionPlugin(Plugins.Base):
    thread = None
    audio_stream = None
    continue_recording = True

    needs_sample_rate_conversion = False
    py_audio = None

    prev_ema_left = 0
    prev_ema_right = 0
    prev_ema_average = 0

    sample_rate = 44100

    def exponential_moving_average(self, current_value, previous_ema, smoothing_factor):
        inverted_smoothing_factor = 1 - smoothing_factor
        return (current_value * inverted_smoothing_factor) + (previous_ema * smoothing_factor)

    def get_audio_amplitude(self, audio_data, low_freq=None, high_freq=None):
        if low_freq or high_freq:
            audio_data = self.filter_frequency(audio_data, low_freq, high_freq)
        peak_amplitude = np.max(np.abs(audio_data))
        return peak_amplitude

    def audio_to_mono(self, audio_data, channels, down_sample_method='average'):
        if channels > 1:
            mono_data = None
            # Reshape the array into a 2D array with two columns (one for each channel)
            audio_data = audio_data.reshape(-1, channels)
            if down_sample_method == 'average':
                # Average the two channels, and convert back to int16
                mono_data = audio_data.mean(axis=1, dtype=np.int16)
            elif down_sample_method == 'left':
                mono_data = audio_data[:, 0]
            elif down_sample_method == 'right':
                mono_data = audio_data[:, 1]
            return mono_data
        else:
            return audio_data

    def normalize_value(self, value, max_value=32767):
        gain_setting = self.get_plugin_setting("gain", 0.8)
        if not isinstance(gain_setting, float) and not isinstance(gain_setting, int):
            return value / max_value
        return (value / max_value) * gain_setting

    def clamp_float(self, value, min_value=0, max_value=1):
        return max(min(value, max_value), min_value)

    def calculate_audio_direction(self, left_amplitude, right_amplitude):
        if left_amplitude == 0 and right_amplitude == 0:
            return 0.5
        else:
            return self.clamp_float(((-1 * left_amplitude) * 2) + (right_amplitude * 2) + 0.5)

    def start_audio_device(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        #SAMPLE_RATE = 16000
        CHUNK = int(self.sample_rate / 10)

        device_index = int(self.get_plugin_setting("loopback_device_index", settings.GetOption("device_out_index")))
        if device_index is None or device_index == -1:
            device_index = settings.GetOption("device_default_out_index")

        py_audio = pyaudio.PyAudio()

        print("Using device index: " + str(device_index))

        try:
            stream, self.needs_sample_rate_conversion, recorded_sample_rate, is_mono = audio_tools.start_recording_audio_stream(
                device_index,
                sample_format=FORMAT,
                sample_rate=self.sample_rate,
                channels=CHANNELS,
                chunk=CHUNK,
                py_audio=py_audio,
            )

            return stream

        except Exception as e:
            print(e)
            return None

    def lerp(self, a: float, b: float, t: float) -> float:
        """Linear interpolate on the scale given by a to b, using t as the point on that scale.
        Examples
        --------
            50 == lerp(0, 100, 0.5)
            4.2 == lerp(1, 5, 0.8)
        """
        return (1 - t) * a + t * b

    def inv_lerp(self, a: float, b: float, v: float) -> float:
        """Inverse Linar Interpolation, get the fraction between a and b on which v resides.
        Examples
        --------
            0.5 == inv_lerp(0, 100, 50)
            0.8 == inv_lerp(1, 5, 4.2)
        """
        return (v - a) / (b - a)

    def remap(self, i_min: float, i_max: float, o_min: float, o_max: float, v: float) -> float:
        """Remap values from one linear scale to another, a combination of lerp and inv_lerp.
        i_min and i_max are the scale on which the original value resides,
        o_min and o_max are the scale to which it should be mapped.
        Examples
        --------
            45 == remap(0, 100, 40, 50, 50)
            6.2 == remap(1, 5, 3, 7, 4.2)
        """
        return self.lerp(o_min, o_max, self.inv_lerp(i_min, i_max, v))


    def filter_frequency(self, audio_data, low_freq, high_freq):
        # Convert audio data to frequency domain using FFT
        audio_freq = np.fft.rfft(audio_data)

        # Zero out unwanted frequencies
        frequencies = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
        if low_freq:
            audio_freq[frequencies < low_freq] = 0
        if high_freq:
            audio_freq[frequencies > high_freq] = 0

        # Convert back to time domain using IFFT
        filtered_audio = np.fft.irfft(audio_freq)
        return filtered_audio

    def audio_loop(self):
        print(self.__class__.__name__ + " thread is started.")
        osc_ip = settings.GetOption("osc_ip")
        osc_port = settings.GetOption("osc_port")

        device_index = int(self.get_plugin_setting("loopback_device_index", settings.GetOption("device_out_index")))
        if device_index is None or device_index == -1:
            device_index = settings.GetOption("device_default_out_index")

        dev_info = self.py_audio.get_device_info_by_index(device_index)
        recorded_sample_rate = int(dev_info['defaultSampleRate'])

        self.continue_recording = True
        while self.continue_recording:
            audio_chunk = self.audio_stream.read(self.get_plugin_setting("num_samples", 32),
                                                 exception_on_overflow=False)
            if self.needs_sample_rate_conversion:
                audio_chunk = audio_tools.resampy_audio(audio_chunk, recorded_sample_rate, self.sample_rate, 2,
                                                         is_mono=False).tobytes()
            audio_int16 = np.frombuffer(audio_chunk, np.int16)

            mono_left_audio = self.audio_to_mono(audio_int16, 2, 'left')
            mono_right_audio = self.audio_to_mono(audio_int16, 2, 'right')

            low_freq = None
            high_freq = None
            if self.get_plugin_setting("bass_filter"):
                low_freq = self.get_plugin_setting("bass_min_frequency")
                high_freq = self.get_plugin_setting("bass_max_frequency")

            peak_amplitude_left = self.get_audio_amplitude(mono_left_audio, low_freq, high_freq)
            peak_amplitude_right = self.get_audio_amplitude(mono_right_audio, low_freq, high_freq)

            if self.get_plugin_setting("debug", False):
                print("peak_amplitude_left: " + str(peak_amplitude_left))
                print("peak_amplitude_right: " + str(peak_amplitude_right))

            # self.prev_ema_left = self.lerp(self.prev_ema_left, peak_amplitude_left, self.get_plugin_setting("smoothing_factor", 0.3))
            # self.prev_ema_right = self.lerp(self.prev_ema_right, peak_amplitude_right, self.get_plugin_setting("smoothing_factor", 0.3))
            self.prev_ema_left = self.exponential_moving_average(peak_amplitude_left, self.prev_ema_left,
                                                                 self.get_plugin_setting("smoothing_factor", 0.3))
            self.prev_ema_right = self.exponential_moving_average(peak_amplitude_right, self.prev_ema_right,
                                                                  self.get_plugin_setting("smoothing_factor", 0.3))

            normalized_amplitude_left = self.normalize_value(self.prev_ema_left)
            normalized_amplitude_right = self.normalize_value(self.prev_ema_right)

            # Boost volume to usable level
            normalized_amplitude_left *= 10
            normalized_amplitude_right *= 10

            audio_volume = self.clamp_float((normalized_amplitude_left + normalized_amplitude_right) / 2)
            audio_direction = self.calculate_audio_direction(normalized_amplitude_left, normalized_amplitude_right)

            VRC_OSCLib.Float(audio_volume, "/avatar/parameters/audio_volume", osc_ip, osc_port)
            VRC_OSCLib.Float(audio_direction, "/avatar/parameters/audio_direction", osc_ip, osc_port)

            if self.get_plugin_setting("debug", False):
                print("Audio volume: " + str(audio_volume))
                print("Audio direction: " + str(audio_direction))

            if not self.continue_recording:
                print(self.__class__.__name__ + " thread is stopped.")
                break

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                #"loopback_device_index": settings.GetOption("device_out_index"),
                "loopback_device_index": {"type": "select_audio", "device_type": "input", "value": str(settings.GetOption("device_index"))},
                "debug": False,
                "gain": {"type": "slider", "min": 0.0, "max": 2.0, "step": 0.05, "value": 0.8},
                "smoothing_factor": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.3},  # A value between 0 and 1, higher value means less smoothing
                "num_samples": 4000,  # Number of samples to read at a time
                "bass_filter": False,
                "bass_min_frequency": 20,
                "bass_max_frequency": 250
            },
            settings_groups={
                "General": ["gain", "smoothing_factor", "bass_filter", "bass_min_frequency", "bass_max_frequency"],
                "Audio Settings": ["loopback_device_index", "debug", "num_samples"]
            }
        )

        if self.is_enabled(False):
            self.py_audio = pyaudio.PyAudio()

            self.audio_stream = self.start_audio_device()
            if self.thread is None and self.audio_stream is not None:
                self.thread = threading.Thread(target=self.audio_loop)
                self.thread.start()

        else:
            if self.thread is not None:
                self.continue_recording = False

                self.thread.join()
                self.thread = None
            if self.audio_stream is not None:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        self.init()
        pass
