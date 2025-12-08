# ============================================================
# Audiobooks Plugin for Whispering Tiger
# V0.0.1
# Generate audiobooks from text files using TTS
# Whispering Tiger: https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import io
import json
import os
import re
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
from kaldiio.wavio import write_wav

import Plugins
import audio_tools
import websocket
from Models.TTS.chatterbox_tts import Chatterbox
import Models.STT.faster_whisper as faster_whisper
from Models.TTS.kokoro_tts import KokoroTTS


class AudiobooksPlugin(Plugins.Base):
    model = None
    transcript_model = None

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "audiobook_file": {"type": "file_open", "accept": ".txt", "value": ""},
                "project_folder": {"type": "folder_open", "accept": "", "value": ""},
                "generate_button": {"label": "Generate Audiobook", "type": "button", "style": "default"},
                "zzz_general_info": {
                    "label": "Supported TTS Models: Chatterbox, Kokoro TTS.\nSupported STT Model: Faster Whisper.",
                    "type": "label", "style": "center"},

                "enable_validation": True,
                "max_retry_attempts": {"type": "slider", "min": 0, "max": 20, "step": 1, "value": 4},
                "validate_max_distance_threshold": {"type": "slider", "min": 0, "max": 20, "step": 1, "value": 3},
                "validate_max_word_num_difference": {"type": "slider", "min": 0, "max": 20, "step": 1, "value": 1},
                "zzz_validation_info": {
                    "label": "Requires the 'Faster Whisper' Speech-to-Text model to be configured and loaded.\nValidation not supported with Kokoro TTS.",
                    "type": "label", "style": "center"},
            },
            settings_groups={
                "General": ["audiobook_file", "project_folder", "generate_button", "zzz_general_info"],
                "Validation": ["enable_validation", "max_retry_attempts", "validate_max_distance_threshold", "validate_max_word_num_difference", "zzz_validation_info"],
            }
        )

    def _levenshtein_distance(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def _search_word_levenshtein(self, original_text, generated_text, threshold=2, max_word_num_difference=0):
        """Returns True if the average minimum Levenshtein distance per word is within the threshold."""
        # Remove punctuation
        for char in [".", ",", "!", "?", ":", ";", "#", "(", ")", "[", "]", "{", "}", "\"", "´", "`", "-", "—", "_", "=", "+", "*", "/", "\\", "|", "<", ">", "\n", "\t", "\r", "！", "。", "？", "！"]:
            generated_text = generated_text.replace(char, " ")
            original_text = original_text.replace(char, " ")

        # Handle apostrophes: remove only if not between letters (e.g., start/end of line)
        generated_text = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", " ", generated_text)
        original_text = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", " ", original_text)

        # Convert to lowercase and split the text into words while removing extra spaces
        generated_text_words = [word for word in generated_text.lower().split()]
        original_text_words = [word for word in original_text.lower().split()]

        #print(f"Original words: {original_text_words}")
        #print(f"Generated words: {generated_text_words}")

        total_min_distance = 0
        num_words = len(original_text_words)

        # Calculate difference of number of words
        word_count_difference = abs(len(generated_text_words) - len(original_text_words))
        if word_count_difference > max_word_num_difference:
            return False

        # Search for the best match for each word in the original text
        for single_orig_word in original_text_words:
            word_min_distance = float('inf')  # Start with a large number

            # Compare with each word in the generated text
            for single_gen_word in generated_text_words:
                current_distance = self._levenshtein_distance(single_gen_word, single_orig_word)
                if current_distance < word_min_distance:
                    word_min_distance = current_distance

            # Aggregate the minimum distance for each word
            total_min_distance += word_min_distance

        # Calculate average minimum distance per word
        average_min_distance = total_min_distance / num_words if num_words > 0 else float('inf')

        # Compare average minimum distance to the threshold
        return average_min_distance <= threshold

    def _read_audiobook_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except Exception as e:
            print(f"Error reading audiobook file: {e}")
            return ""


    def transcribe(self, audio_data):
        compute_dtype = self._settings.GetOption("whisper_precision")
        model = self._settings.GetOption("model")
        ai_device = self._settings.GetOption("ai_device")
        whisper_language = self.get_plugin_setting("language_spoken")
        if whisper_language == "auto":
            whisper_language = None

        if self.transcript_model is None:
            self.transcript_model = faster_whisper.FasterWhisper(model, device=ai_device, compute_type=compute_dtype,
                                                                 cpu_threads=2, num_workers=2)

        if self.transcript_model is not None:
            try:
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.detach().cpu().numpy()
                    audio_data = audio_tools.convert_audio_datatype_to_integer(audio_data)
                audio_data = audio_tools.resample_audio(audio_data, recorded_sample_rate=self.model.sample_rate, target_sample_rate=16_000, target_channels=1, input_channels=1, dtype="int16")

                #audio_data = np.int16(audio_data * 32767)  # Convert to 16-bit PCM
                # buff = io.BytesIO()
                # write_wav(buff, self.model.sample_rate, audio_data)
                # buff.seek(0)

                result = self.transcript_model.transcribe(audio_data, task="transcribe",
                                                          language=whisper_language,
                                                          condition_on_previous_text=True,
                                                          initial_prompt=None,
                                                          logprob_threshold=-1.0,
                                                          no_speech_threshold=0.6,
                                                          temperature=[
                                                              0.0,
                                                              0.2,
                                                              0.4,
                                                              0.6,
                                                              0.8,
                                                              1.0,
                                                          ],
                                                          beam_size=5,
                                                          word_timestamps=False,
                                                          without_timestamps=False,
                                                          patience=1.0
                                                          )
                if result is not None and 'text' in result:
                    return result.get('text').strip()
                else:
                    return ""

            except Exception as e:
                print(e)
                return ""
        return ""


    def generate_audiobook(self):
        audiobook_file = self.get_plugin_setting("audiobook_file")
        project_folder = self.get_plugin_setting("project_folder")
        project_file = os.path.join(project_folder, "audiobook_output.wav")
        enable_validation = self.get_plugin_setting("enable_validation")

        tts_model_type = self._settings.GetOption("tts_type")
        print(f"TTS requested {tts_model_type} (Audiobook)")
        if self.model is None:
            if tts_model_type == "chatterbox":
                self.model = Chatterbox()
            elif tts_model_type == "kokoro":
                self.model = KokoroTTS()
                enable_validation = False  # Disable validation for Kokoro TTS
            else:
                print(f"TTS model type {tts_model_type} is not supported for audiobook generation.")
                websocket.BroadcastMessage(
                    json.dumps({"type": "info", "data": f"TTS model type {tts_model_type} is not supported for audiobook generation."}))
                return
        self.model._ensure_special_settings()

        ref_audio = None
        max_retry_attempts = self.get_plugin_setting("max_retry_attempts")
        levenshtein_threshold = int(self.get_plugin_setting("validate_max_distance_threshold"))
        max_word_num_difference = int(self.get_plugin_setting("validate_max_word_num_difference"))

        if not audiobook_file or not project_folder:
            print("Please specify both audiobook file and project folder.")
            websocket.BroadcastMessage(
                json.dumps({"type": "info", "data": "Please specify both audiobook file and project folder."}))
            return

        if Path(project_file).exists():
            print(f"Output file already exists in {project_folder}. Please choose a different project folder.")
            websocket.BroadcastMessage(
                json.dumps({"type": "info", "data": "Output file already exists in the project folder. Please choose a different project folder."}))
            return

        text = self._read_audiobook_file(audiobook_file)

        writer = audio_tools.WavWriter(project_file, self.model.sample_rate, 1, sample_format="float32")

        if not hasattr(self.model, 'stream_tts_segments'):
            print("The current TTS model does not support streaming TTS segments.")
            websocket.BroadcastMessage(
                json.dumps({"type": "info", "data": "The current TTS model does not support streaming TTS segments."}))
            return

        try:
            # Stream segments and write raw PCM into the WAV container (wav is torch.Tensor)
            for wav, voice_name, inserted_silence, section_info in self.model.stream_tts_segments(
                    text, ref_audio, regenerate_section=None
            ):
                # Only count actual speech segments that have associated text; skip pure silences.
                is_speech_segment = section_info is not None and "text" in section_info and section_info["text"].strip() != ""

                if enable_validation and max_retry_attempts > 0 and is_speech_segment:
                    segment_text = section_info["text"]
                    #print(f"Validating transcription for segment: '{segment_text}'")

                    generated_text = self.transcribe(wav)
                    #print(f"Generated transcription: '{generated_text}'")

                    levenshtein_text_valid = self._search_word_levenshtein(segment_text, generated_text, levenshtein_threshold, max_word_num_difference)
                    #print(f"Levenshtein validation result: {levenshtein_text_valid}")

                    attempts = 0
                    while not levenshtein_text_valid and attempts < max_retry_attempts:
                        print(f"Transcription validation failed. Retrying generation for segment: '{segment_text}'")
                        wav, voice_name, inserted_silence, section_info = next(self.model.stream_tts_segments(
                            text, ref_audio, regenerate_section=section_info
                        ))
                        generated_text = self.transcribe(wav)
                        levenshtein_text_valid = self._search_word_levenshtein(segment_text, generated_text, levenshtein_threshold, max_word_num_difference)
                        print(f"Levenshtein Re-validation result: {levenshtein_text_valid}")
                        attempts += 1

                wav_bytes = self.model.return_pcm_audio(wav)

                if wav_bytes and len(wav_bytes) > 0:
                    writer.write_frames(wav_bytes)

            writer.close()
            print("100% Finished. file saved to:", project_file)
            websocket.BroadcastMessage(
                json.dumps({"type": "info", "data": "Generation complete!"}))

        except Exception as e:
            writer.close()
            print("Failed to save file:", e)
            traceback.print_exc()


    # def generate_audiobook(self):
    #     if self.model is None:
    #         self.model = Chatterbox()
    #     audiobook_file = self.get_plugin_setting("audiobook_file")
    #     project_folder = self.get_plugin_setting("project_folder")
    #
    #     if not audiobook_file or not project_folder:
    #         print("Please specify both audiobook file and project folder.")
    #         websocket.BroadcastMessage(
    #             json.dumps({"type": "info", "data": "Please specify both audiobook file and project folder."}))
    #         return
    #
    #     if Path(project_folder, "audiobook_output.wav").exists():
    #         print(f"Output file already exists in {project_folder}. Please choose a different project folder.")
    #         websocket.BroadcastMessage(
    #             json.dumps({"type": "info", "data": "Output file already exists in the project folder. Please choose a different project folder."}))
    #         return
    #
    #     print(f"Generating audiobook from {audiobook_file} into {project_folder}.")
    #     text = self._read_audiobook_file(audiobook_file)
    #     self.model.generate_audiobook(text, None, project_folder)
    #
    #     websocket.BroadcastMessage(
    #         json.dumps({"type": "info", "data": "Generation complete!"}))

    def on_event_received(self, message, websocket_connection=None):
        if self.is_enabled(False):
            if "type" not in message:
                return
            if message["type"] == "plugin_button_press":
                if message["value"] == "generate_button":
                    self.generation_thread = threading.Thread(target=self.generate_audiobook, daemon=True)
                    self.generation_thread.start()
