# ============================================================
# Subtitles Export Plugin for Whispering Tiger
# V0.0.5
# See https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import json
import csv
import os

from faster_whisper.transcribe import Segment, TranscriptionInfo

import Plugins
import settings
import websocket

import Models.STT.faster_whisper as faster_whisper
from Models.TextTranslation import texttranslate
from whisper.tokenizer import LANGUAGES
from typing import TextIO, cast, Tuple, Iterable


class SubtitleExportPlugin(Plugins.Base):
    audio_model = None

    def init(self):
        whisper_languages = sorted(LANGUAGES.keys())
        whisper_languages.insert(0, "auto")
        text_translation_languages = []
        texttranslate_languages = texttranslate.GetInstalledLanguageNames()
        if texttranslate_languages is not None:
            text_translation_languages = [lang['code'] for lang in texttranslate_languages]
        source_text_translation_languages = list(text_translation_languages)
        text_translation_languages.insert(0, "")

        source_text_translation_languages.insert(0, "auto-by-text")
        source_text_translation_languages.insert(0, "auto-by-speech")
        source_text_translation_languages.insert(0, "")

        # prepare all possible settings
        self.init_plugin_settings(
            {
                # General
                "language_spoken": {"type": "select", "value": "auto", "values": whisper_languages},
                "language_txt_transcript": {"type": "select", "value": "", "values": source_text_translation_languages},
                "language_target": {"type": "select", "value": "", "values": text_translation_languages},
                "language_z_info_label": {"type": "label", "label": "set \"language_txt_transcript\" to empty to use same as \"language_spoken\",\nor to \"auto-*\" to detect the language\n(useful for multiple languages in the audio)", "style": "center"},
                "audio_filepath": {"type": "file_open", "accept": ".wav,.mp3,.mp4", "value": "audio.wav"},
                "subtitle_file": {"type": "file_save", "accept": ".srt,.vtt,.sbv,.csv,.txt", "value": "subtitle.srt"},
                "subtitle_file_label": {"type": "label", "label": "Supported subtitle types are .srt, .sbv, .vtt, .csv or .txt", "style": "center"},
                "z_transcribe_button": {"label": "Start Transcription", "type": "button", "style": "primary"},
            },
            settings_groups={
                "General": ["audio_filepath", "z_transcribe_button", "language_spoken", "language_txt_transcript", "language_target", "language_z_info_label", "subtitle_file", "subtitle_file_label"],
            }
        )

    def _format_timestamp(self, seconds: float, decimal_marker: str = "."):
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

    def iterate_result(self, result: Tuple[Iterable[Segment], TranscriptionInfo], decimal_marker: str = "."):
        for segment in result:
            segment_start = self.format_timestamp(cast(float, segment.start), decimal_marker)
            segment_end = self.format_timestamp(cast(float, segment.end), decimal_marker)
            segment_text = segment.text.strip()
            yield segment_start, segment_end, segment_text

    def format_timestamp(self, seconds: float, decimal_marker: str = "."):
        return self._format_timestamp(
            seconds=seconds,
            decimal_marker=decimal_marker,
        )

    def write_srt(self, result):
        write_file = self.get_plugin_setting("subtitle_file")
        decimal_marker: str = ","

        def _write_result(write_result: Tuple[Iterable[Segment], TranscriptionInfo], file: TextIO):
            for i, (start, end, text) in enumerate(self.iterate_result(write_result, decimal_marker), start=1):
                print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)

        with open(write_file, "w", encoding="utf-8") as f:
            _write_result(result, f)

    def write_sbv(self, result):
        write_file = self.get_plugin_setting("subtitle_file")
        decimal_marker: str = "."

        def _write_result(write_result: Tuple[Iterable[Segment], TranscriptionInfo], file: TextIO):
            for i, (start, end, text) in enumerate(self.iterate_result(write_result, decimal_marker), start=1):
                print(f"{start},{end}\n{text}\n", file=file, flush=True)

        with open(write_file, "w", encoding="utf-8") as f:
            _write_result(result, f)

    def write_vtt(self, result):
        write_file = self.get_plugin_setting("subtitle_file")
        decimal_marker: str = "."

        def _write_result(write_result: Tuple[Iterable[Segment], TranscriptionInfo], file: TextIO):
            print("WEBVTT\n", file=file)
            for start, end, text in self.iterate_result(write_result, decimal_marker):
                print(f"{start} --> {end}\n{text}\n", file=file, flush=True)

        with open(write_file, "w", encoding="utf-8") as f:
            _write_result(result, f)

    def write_csv(self, result):
        write_file = self.get_plugin_setting("subtitle_file")
        decimal_marker: str = ","

        def _write_result(write_result: Tuple[Iterable[Segment], TranscriptionInfo], csv_writer):
            for start, end, text in self.iterate_result(write_result, decimal_marker):
                csv_writer.writerow([start, end, text])

        with open(write_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Start', 'End', 'Text'])  # Optional: Write header
            _write_result(result, writer)

    def write_txt(self, result):
        write_file = self.get_plugin_setting("subtitle_file")

        def _write_result(write_result: Tuple[Iterable[Segment], TranscriptionInfo], file: TextIO):
            for _, _, text in self.iterate_result(write_result):
                print(text, file=file)

        with open(write_file, "w", encoding="utf-8") as f:
            _write_result(result, f)

    def transcribe(self):
        compute_dtype = settings.GetOption("whisper_precision")
        model = settings.GetOption("model")
        ai_device = settings.GetOption("ai_device")
        audio_filepath = self.get_plugin_setting("audio_filepath")
        write_file = self.get_plugin_setting("subtitle_file")
        whisper_language = self.get_plugin_setting("language_spoken")
        text_language = self.get_plugin_setting("language_txt_transcript")
        if whisper_language == "auto":
            whisper_language = None
        target_language = self.get_plugin_setting("language_target")

        file_extension = os.path.splitext(write_file)[-1].lower()

        # Mapping of file extensions to corresponding methods
        extension_to_method = {
            ".srt": self.write_srt,
            ".vtt": self.write_vtt,
            ".sbv": self.write_sbv,
            ".csv": self.write_csv,
            ".txt": self.write_txt
        }
        # Check if the file extension is supported and get the corresponding method
        write_method = extension_to_method.get(file_extension)

        if write_method is None:
            # Handle the case for unsupported file extension
            print(f"Unsupported file extension: {file_extension}")
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": f"Unsupported file extension: {file_extension}"}))
            return  # Exit the function

        if self.audio_model is None:
            self.audio_model = faster_whisper.FasterWhisper(model, device=ai_device, compute_type=compute_dtype,
                                                            cpu_threads=2, num_workers=2)

        if self.audio_model is not None:
            websocket.set_loading_state("subtitle_export_plugin_working", True)

            try:
                result_segments, audio_info = self.audio_model.model.transcribe(audio_filepath, task="transcribe",
                                                                                language=whisper_language,
                                                                                condition_on_previous_text=True,
                                                                                initial_prompt=None,
                                                                                log_prob_threshold=-1.0,
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
                                                                                #word_timestamps=True,
                                                                                word_timestamps=False,
                                                                                without_timestamps=False,
                                                                                patience=1.0
                                                                                )
                if target_language != "":
                    translated_segments = []

                    if text_language == "" or text_language == "auto-by-speech" or text_language == "auto-by-text":
                        if text_language == "auto-by-speech" and whisper_language is None:
                            text_language = audio_info.language
                        elif text_language == "auto-by-text" and whisper_language is None:
                            text_language = "auto"
                        else:
                            text_language = whisper_language

                        # get ISO3 language code
                        if whisper_language in texttranslate.texttranslateNLLB200_CTranslate2.LANGUAGES_ISO1_TO_ISO3:
                            text_language = texttranslate.texttranslateNLLB200_CTranslate2.LANGUAGES_ISO1_TO_ISO3[whisper_language][0]

                    for segment in result_segments:
                        translation, _, _ = texttranslate.TranslateLanguage(
                            segment.text,
                            text_language,
                            target_language,
                            as_iso1=False
                        )
                        translated_segments.append(Segment(
                            id=segment.id,
                            seek=segment.seek,
                            start=segment.start,
                            end=segment.end,
                            text=translation,
                            temperature=segment.temperature,
                            avg_logprob=segment.avg_logprob,
                            no_speech_prob=segment.no_speech_prob,
                            compression_ratio=segment.compression_ratio,
                            tokens=segment.tokens,
                            words=segment.words
                        ))
                    result_segments = translated_segments

                # Call the method associated with the file extension
                write_method(result_segments)

            except Exception as e:
                print(e)
                websocket.BroadcastMessage(json.dumps({"type": "error", "data": "Generating subtitle file failed:\n" + str(e)}))

            websocket.set_loading_state("subtitle_export_plugin_working", False)

            websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Generating subtitle file finished."}))

    def on_event_received(self, message, websocket_connection=None):
        if self.is_enabled(False):
            if "type" not in message:
                return
            if message["type"] == "plugin_button_press":
                if message["value"] == "z_transcribe_button":
                    self.transcribe()
        else:
            websocket.BroadcastMessage(json.dumps({"type": "info", "data": "Plugin is disabled."}))
