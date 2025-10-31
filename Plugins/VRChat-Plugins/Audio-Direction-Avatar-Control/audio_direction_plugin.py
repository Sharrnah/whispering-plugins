# ============================================================
# Sends volume and audio direction over OSC using Whispering Tiger
# Version 1.1.2
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

    # Track input characteristics
    input_channel_num = 2
    recorded_sample_rate = sample_rate

    def exponential_moving_average(self, current_value, previous_ema, smoothing_factor):
        inverted_smoothing_factor = 1 - smoothing_factor
        return (current_value * inverted_smoothing_factor) + (previous_ema * smoothing_factor)

    def get_audio_amplitude(self, audio_data, low_freq=None, high_freq=None):
        if low_freq or high_freq:
            audio_data = self.filter_frequency(audio_data, low_freq, high_freq)
        # Guard against empty arrays
        if audio_data is None or len(audio_data) == 0:
            return 0.0
        peak_amplitude = np.max(np.abs(audio_data))
        return float(peak_amplitude)

    def audio_to_mono(self, audio_data, channels, down_sample_method='average'):
        if channels > 1:
            mono_data = None
            # Reshape the array into a 2D array with one column per channel
            try:
                audio_data = audio_data.reshape(-1, channels)
            except Exception:
                # Fallback: if reshape fails due to buffer length mismatch, trim to the nearest full frame
                frame_count = (len(audio_data) // channels)
                audio_data = audio_data[: frame_count * channels].reshape(-1, channels)
            if down_sample_method == 'average':
                # Average the channels in float to avoid integer truncation, then cast back to int16
                mono_data = audio_data.mean(axis=1).astype(np.int16)
            elif down_sample_method == 'left':
                mono_data = audio_data[:, 0]
            elif down_sample_method == 'right':
                mono_data = audio_data[:, 1]
            else:
                mono_data = audio_data.mean(axis=1).astype(np.int16)
            return mono_data
        else:
            return audio_data

    def normalize_value(self, value, max_value=32767):
        gain_setting = self.get_plugin_setting("gain", 0.8)
        if not isinstance(gain_setting, float) and not isinstance(gain_setting, int):
            return value / max_value
        return (value / max_value) * float(gain_setting)

    def clamp_float(self, value, min_value=0, max_value=1):
        return max(min(value, max_value), min_value)

    def calculate_audio_direction(self, left_amplitude, right_amplitude):
        # Robust, ratio-based mapping: 0.0 fully left, 1.0 fully right, 0.5 centered
        total = left_amplitude + right_amplitude
        if total <= 1e-9:
            return 0.5
        direction = right_amplitude / total
        return self.clamp_float(direction, 0, 1)

    def start_audio_device(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        CHUNK = int(self.sample_rate / 10)

        device_index = int(self.get_plugin_setting("loopback_device_index", settings.GetOption("device_out_index")))
        if device_index is None or device_index == -1:
            device_index = settings.GetOption("device_default_out_index")

        # Reuse the instance created in init
        if self.py_audio is None:
            self.py_audio = pyaudio.PyAudio()

        print("Using device index: " + str(device_index))

        try:
            stream, needs_conv, recorded_sample_rate, input_channels = audio_tools.start_recording_audio_stream(
                device_index,
                sample_format=FORMAT,
                sample_rate=self.sample_rate,
                channels=CHANNELS,
                chunk=CHUNK,
                py_audio=self.py_audio,
            )

            # Persist stream characteristics
            self.needs_sample_rate_conversion = needs_conv
            self.recorded_sample_rate = int(recorded_sample_rate)
            # audio_tools returns the number of channels opened
            self.input_channel_num = int(input_channels) if input_channels else 2

            print(f"Audio stream opened: recorded_sample_rate={self.recorded_sample_rate}, channels={self.input_channel_num}, needs_resample={self.needs_sample_rate_conversion}")

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

    def _calc_strength(self, samples: np.ndarray, method: str = "rms") -> float:
        # Accept int16 or float arrays; compute per-chunk strength
        if samples is None or len(samples) == 0:
            return 0.0
        if method == "peak":
            return float(np.max(np.abs(samples)))
        # Default to RMS for stability
        # normalize to float to prevent overflow in squares
        s = samples.astype(np.float32)
        return float(np.sqrt(np.mean(s * s)))

    def _extract_channel(self, audio_data: np.ndarray, channels: int, index: int) -> np.ndarray:
        # Safely extract a single channel from interleaved PCM buffer
        if channels <= 1:
            return audio_data
        try:
            frames = (len(audio_data) // channels)
            arr = audio_data[: frames * channels].reshape(-1, channels)
        except Exception:
            return audio_data  # fallback to raw if reshape fails
        # Clamp index
        idx = max(0, min(index, channels - 1))
        return arr[:, idx]

    def audio_loop(self):
        print(self.__class__.__name__ + " thread is started.")
        osc_ip = settings.GetOption("osc_ip")
        osc_port = settings.GetOption("osc_port")

        device_index = int(self.get_plugin_setting("loopback_device_index", settings.GetOption("device_out_index")))
        if device_index is None or device_index == -1:
            device_index = settings.GetOption("device_default_out_index")

        # We already stored recorded_sample_rate from the opened stream
        self.continue_recording = True
        while self.continue_recording:
            audio_chunk = self.audio_stream.read(self.get_plugin_setting("num_samples", 32),
                                                 exception_on_overflow=False)
            # Resample and enforce stereo target if needed for direction calc
            if self.needs_sample_rate_conversion:
                # If the input was mono, tell the resampler; force stereo out (2 channels)
                audio_chunk = audio_tools.resampy_audio(audio_chunk,
                                                         self.recorded_sample_rate,
                                                         self.sample_rate,
                                                         2,
                                                         is_mono=(self.input_channel_num == 1)).tobytes()

            audio_int16 = np.frombuffer(audio_chunk, np.int16)

            # Determine the channel count of the current buffer
            channels = 2 if self.needs_sample_rate_conversion else max(1, int(self.input_channel_num))

            # Extract per-side mono streams using configured indices
            if channels > 1:
                left_idx = int(self.get_plugin_setting("left_channel_index", 0))
                right_idx = int(self.get_plugin_setting("right_channel_index", 1))
                mono_left_audio = self._extract_channel(audio_int16, channels, left_idx)
                mono_right_audio = self._extract_channel(audio_int16, channels, right_idx)
            else:
                # Mono input: both sides see the same signal; direction should be centered
                mono_left_audio = audio_int16
                mono_right_audio = audio_int16

            low_freq = None
            high_freq = None
            if self.get_plugin_setting("bass_filter"):
                low_freq = self.get_plugin_setting("bass_min_frequency")
                high_freq = self.get_plugin_setting("bass_max_frequency")

            # Choose measurement method
            meter_method = self.get_plugin_setting("meter_method", "rms")

            # Compute raw strengths
            if low_freq or high_freq:
                mono_left_proc = self.filter_frequency(mono_left_audio, low_freq, high_freq)
                mono_right_proc = self.filter_frequency(mono_right_audio, low_freq, high_freq)
            else:
                mono_left_proc = mono_left_audio
                mono_right_proc = mono_right_audio

            raw_left = self._calc_strength(mono_left_proc, meter_method)
            raw_right = self._calc_strength(mono_right_proc, meter_method)

            # Per-channel calibration compensation (to correct any inherent gain imbalance)
            cal_left = float(self.get_plugin_setting("calibration_left", 1.0))
            cal_right = float(self.get_plugin_setting("calibration_right", 1.0))
            raw_left *= cal_left
            raw_right *= cal_right

            if self.get_plugin_setting("debug", False):
                print(f"raw_strength_left({meter_method}): {raw_left}")
                print(f"raw_strength_right({meter_method}): {raw_right}")

            # Smooth
            self.prev_ema_left = self.exponential_moving_average(raw_left, self.prev_ema_left,
                                                                 self.get_plugin_setting("smoothing_factor", 0.3))
            self.prev_ema_right = self.exponential_moving_average(raw_right, self.prev_ema_right,
                                                                  self.get_plugin_setting("smoothing_factor", 0.3))

            # Normalize to [0..1] (with gain) for direction and volume calculations
            norm_left = self.normalize_value(self.prev_ema_left)
            norm_right = self.normalize_value(self.prev_ema_right)

            # Optional gate and shaping before direction
            floor_gate = float(self.get_plugin_setting("floor_gate", 0.0))
            shaped_left = max(0.0, norm_left - floor_gate)
            shaped_right = max(0.0, norm_right - floor_gate)
            gamma = float(self.get_plugin_setting("direction_gamma", 1.0))
            if gamma != 1.0:
                shaped_left = shaped_left ** gamma
                shaped_right = shaped_right ** gamma

            # Optional channel swap
            if self.get_plugin_setting("swap_channels", False):
                shaped_left, shaped_right = shaped_right, shaped_left

            # Direction should be computed from normalized but unclipped values (no extra boost)
            audio_direction = self.calculate_audio_direction(shaped_left, shaped_right)

            # Optional invert for avatar param orientation
            if self.get_plugin_setting("invert_direction", False):
                audio_direction = 1.0 - audio_direction

            # Compute a usable volume meter: boost and clamp
            vol_left = self.clamp_float(norm_left * 10.0)
            vol_right = self.clamp_float(norm_right * 10.0)
            audio_volume = self.clamp_float((vol_left + vol_right) / 2.0)

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
                "bass_max_frequency": 250,
                # New options to address channel/device peculiarities
                "swap_channels": False,
                "invert_direction": False,
                # Metering & calibration
                "meter_method": {"type": "select_textvalue", "values": [["RMS","rms"],["Peak","peak"]], "value": "RMS", "_value_real": "rms"},
                "calibration_left": {"type": "slider", "min": 0.5, "max": 1.5, "step": 0.01, "value": 1.0},
                "calibration_right": {"type": "slider", "min": 0.5, "max": 1.5, "step": 0.01, "value": 1.0},
                # Channel selection and direction shaping
                "left_channel_index": {"type": "slider", "min": 0, "max": 7, "step": 1, "value": 0},
                "right_channel_index": {"type": "slider", "min": 0, "max": 7, "step": 1, "value": 1},
                "floor_gate": {"type": "slider", "min": 0.0, "max": 0.2, "step": 0.005, "value": 0.0},
                "direction_gamma": {"type": "slider", "min": 0.5, "max": 3.0, "step": 0.05, "value": 1.0},
            },
            settings_groups={
                "General": ["gain", "smoothing_factor", "bass_filter", "bass_min_frequency", "bass_max_frequency", "meter_method", "calibration_left", "calibration_right", "floor_gate", "direction_gamma"],
                "Channel Mapping": ["left_channel_index", "right_channel_index", "swap_channels", "invert_direction"],
                "Audio Settings": ["loopback_device_index", "debug", "num_samples"]
            }
        )

        if self.is_enabled(False):
            if self.py_audio is None:
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
