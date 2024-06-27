# ============================================================
# Write Transcript Plugin for Whispering Tiger
# V0.0.1
# See https://github.com/Sharrnah/whispering-ui
# ============================================================
#

import Plugins


class WriteTranscriptPlugin(Plugins.Base):
    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                # General
                "transcript_file": {"type": "file_save", "accept": ".txt", "value": "transcript.txt"},
                "write_intermediate_transcript": False,
            },
            settings_groups={
                "General": ["transcript_file", "write_intermediate_transcript"],
            }
        )

    def write_file(self, text):
        with open(self.get_plugin_setting("transcript_file"), "a", newline='') as f:
            f.write(text + "\n")

    def stt(self, text, result_obj):
        if self.is_enabled(False) and text.strip() != "":
            self.write_file(text)
        return

    def stt_intermediate(self, text, result_obj):
        if self.is_enabled(False) and self.get_plugin_setting("write_intermediate_transcript") and text.strip() != "":
            self.write_file(text)
        return
