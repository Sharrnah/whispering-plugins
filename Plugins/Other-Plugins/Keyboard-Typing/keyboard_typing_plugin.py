# ============================================================
# Keyboard Typing Plugin for Whispering Tiger
# V1.0.6
# See https://github.com/Sharrnah/whispering
# ============================================================
#
import threading
import time

import Plugins
from pathlib import Path
import downloader
import os
import sys
from importlib import util

keyboard_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/55/88/287159903c5b3fc6d47b651c7ab65a54dcf9c9916de546188a7f62870d6d/keyboard-0.13.5-py3-none-any.whl",
    "sha256": "8e9c2422f1217e0bd84489b9ecd361027cc78415828f4fe4f88dd4acd587947b",
    "path": "keyboard"
}

keyboard_typing_plugin_dir = Path(Path.cwd() / "Plugins" / "keyboard_typing_plugin")
os.makedirs(keyboard_typing_plugin_dir, exist_ok=True)


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


COMMANDS = {
    "stop typing": {"value": False},
    "start typing": {"value": True},
    "new line": {"value": [13, 10]},
}


class KeyboardTypingPlugin(Plugins.Base):
    commands = None
    keyboard = None
    paused = False

    def levenshtein_distance(self, s1, s2):
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

    def _search_word_levenshtein(self, command_text, threshold=2, word_threshold=1):
        best_match = None
        best_match_score = float('inf')

        # search for command words in command list
        for command in self.commands:
            command_score = 0
            all_words_found = True
            words = command.split(" ")
            for word in words:
                min_word_score = min([self.levenshtein_distance(word, txt_word) for txt_word in command_text])
                if min_word_score > word_threshold:  # You can adjust the threshold for individual words.
                    all_words_found = False
                    break
                command_score += min_word_score
            if not all_words_found:
                continue

            command_score /= len(command)

            if command_score < best_match_score:
                best_match = command
                best_match_score = command_score

        max_distance_threshold = threshold  # You can adjust the threshold to your liking.

        if best_match_score <= max_distance_threshold:
            return True, self.commands[best_match]["value"]

        return False, None

    def command_handler(self, command_text):
        levenshtein_threshold = self.get_plugin_setting("levenshtein_max_distance_threshold", 1)
        levenshtein_word_threshold = self.get_plugin_setting("levenshtein_max_distance_word_threshold", 1)
        self.commands = self.get_plugin_setting("commands", COMMANDS)

        # remove punctuations
        command_text = command_text.replace(".", "").replace(",", "").replace("!", "").replace("?", "")
        # split command into words
        command_text = command_text.strip().lower().split(" ")
        # remove empty strings
        command_text = list(filter(None, command_text))

        found, command_value = self._search_word_levenshtein(command_text, levenshtein_threshold,
                                                                                levenshtein_word_threshold)
        return found, command_value

    def type_custom(self, value):
        # type character
        if isinstance(value, int):
            if not self.paused:
                self.keyboard.write(chr(value))
        # type string
        elif isinstance(value, str):
            if not self.paused:
                self.keyboard.write(value)
        # type list of characters or strings
        elif isinstance(value, list):
            # iterate through character list
            for char in value:
                if not self.paused:
                    if isinstance(char, int):
                        self.keyboard.write(chr(char))
                    elif isinstance(char, str):
                        self.keyboard.write(char)

    def type_text(self, text):
        found, command_value = self.command_handler(text)
        if found:
            self.type_custom(command_value)
            # pause or resume typing
            if isinstance(command_value, bool):
                print("typing paused" if not command_value else "typing resumed")
                self.paused = not command_value
        else:
            auto_chat_enabled = False
            auto_chat_delay_seconds = self.get_plugin_setting("auto_chat_delay_seconds")
            auto_chat_settings = self.get_plugin_setting("auto_chat_options")
            if isinstance(auto_chat_settings, dict) and all(key in auto_chat_settings for key in ["pre_typing_key", "post_typing_key"]):
                auto_chat_enabled = self.get_plugin_setting("auto_chat_enabled")

            if not self.paused:
                if auto_chat_enabled:
                    self.type_custom(auto_chat_settings["pre_typing_key"])
                    time.sleep(auto_chat_delay_seconds)
                self.keyboard.write(text)
                if auto_chat_enabled:
                    time.sleep(auto_chat_delay_seconds)
                    self.type_custom(auto_chat_settings["post_typing_key"])
        return

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "levenshtein_max_distance_threshold": 1,
                "levenshtein_max_distance_word_threshold": 1,

                "auto_chat_enabled": False,
                "auto_chat_delay_seconds": {"type": "slider", "min": 0, "max": 5, "step": 0.01, "value": 0.0},
                "auto_chat_options": {
                    "pre_typing_key": "t",
                    "post_typing_key": [13, 10],
                },
                "commands": COMMANDS,
                "typing_initially_paused": False,
            },
            settings_groups={
                "General": ["commands", "typing_initially_paused"],
                "Advanced": ["levenshtein_max_distance_threshold", "levenshtein_max_distance_word_threshold"],
                "Auto Chat": ["auto_chat_enabled", "auto_chat_options", "auto_chat_delay_seconds"]
            }
        )

        self.paused = self.get_plugin_setting("typing_initially_paused", False)

        auto_chat_settings = self.get_plugin_setting("auto_chat_options")
        if not isinstance(auto_chat_settings, dict) or not all(key in auto_chat_settings for key in ["pre_typing_key", "post_typing_key"]):
            self.set_plugin_setting("auto_chat_options", {
                "pre_typing_key": "t",
                "post_typing_key": [13, 10],
            })

        commands_options = self.get_plugin_setting("commands")
        if not isinstance(commands_options, dict) or commands_options is None or commands_options == "":
            self.set_plugin_setting("commands", COMMANDS)

        # load the keyboard module
        if self.is_enabled(False) and self.keyboard is None:
            if not Path(keyboard_typing_plugin_dir / keyboard_dependency_module["path"] / "__init__.py").is_file():
                downloader.download_extract([keyboard_dependency_module["url"]],
                                            str(keyboard_typing_plugin_dir.resolve()),
                                            keyboard_dependency_module["sha256"],
                                            alt_fallback=True,
                                            fallback_extract_func=downloader.extract_zip,
                                            fallback_extract_func_args=(
                                                str(keyboard_typing_plugin_dir / os.path.basename(keyboard_dependency_module["url"])),
                                                str(keyboard_typing_plugin_dir.resolve()),
                                            ),
                                            title="keyboard module", extract_format="zip")

            self.keyboard = load_module(str(Path(keyboard_typing_plugin_dir / keyboard_dependency_module["path"]).resolve()))
        pass

    def stt(self, text, result_obj):
        if self.is_enabled(False) and text.strip() != "":
            threading.Thread(target=self.type_text, args=(text,)).start()
        return

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        pass
