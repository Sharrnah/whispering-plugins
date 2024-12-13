# ============================================================
# Keyboard Typing Plugin for Whispering Tiger
# V1.0.8
# See https://github.com/Sharrnah/whispering
# ============================================================
#
import platform
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
                # DirectInput keyboard scan codes ( https://gist.github.com/dretax/fe37b8baf55bc30e9d63 )
                if value.startswith('0x') or value.startswith('DI_'):
                    # split at a + to get 2 keys at the same time
                    str_values = value.split("+")
                    # press keys
                    for str_value in str_values:
                        if str_value.startswith('0x'):
                            hex_value = int(str_value, 16)
                            PressKey(hex_value)
                        elif str_value.startswith('DI_'):
                            PressKey(DI_KEY_CODE_LOOKUP[str_value])
                        time.sleep(0.1)

                    # release keys again
                    for str_value in str_values:
                        if str_value.startswith('0x'):
                            hex_value = int(str_value, 16)
                            ReleaseKey(hex_value)
                        elif str_value.startswith('DI_'):
                            ReleaseKey(DI_KEY_CODE_LOOKUP[str_value])

                else:
                    self.keyboard.write(value)
        # type list of characters or strings
        elif isinstance(value, list):
            do_release_keys = False
            # iterate through character list
            for char in value:
                if not self.paused:
                    if isinstance(char, int):
                        self.keyboard.write(chr(char))
                    elif isinstance(char, str):
                        if char.startswith('0x'):
                            do_release_keys = True
                            hex_value = int(char, 16)
                            PressKey(hex_value)
                            time.sleep(0.1)
                        elif char.startswith('DI_'):
                            do_release_keys = True
                            PressKey(DI_KEY_CODE_LOOKUP[char])
                            time.sleep(0.1)
                        else:
                            self.keyboard.write(char)
            if do_release_keys:
                for char in value:
                    if isinstance(char, str):
                        if char.startswith('0x'):
                            hex_value = int(char, 16)
                            ReleaseKey(hex_value)
                        elif char.startswith('DI_'):
                            ReleaseKey(DI_KEY_CODE_LOOKUP[char])

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
                "auto_chat_delay_seconds": {"type": "slider", "min": 0, "max": 5, "step": 0.01, "value": 0.50},
                "auto_chat_options": {
                    "pre_typing_key": "DI_Y",
                    "post_typing_key": ["DI_RETURN", "DI_ESCAPE"],
                },
                "auto_chat_zz_info": {
                    "label": "Use KeyCodes in Hex or with DI_ prefixes for DirectInput key presses., numbers for ASCII codes.\nDirectInput: 'DI_RETURN', 'DI_A' - 'DI_Z' ...,\nuse a list or combine with '+' to press multiple keys at once.",
                    "type": "label", "style": "left"},
                "commands": COMMANDS,
                "typing_initially_paused": False,
            },
            settings_groups={
                "General": ["commands", "typing_initially_paused"],
                "Advanced": ["levenshtein_max_distance_threshold", "levenshtein_max_distance_word_threshold"],
                "Auto Chat": ["auto_chat_enabled", "auto_chat_options", "auto_chat_delay_seconds", "auto_chat_zz_info"]
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

# ====================

if platform.system() == 'Windows':
    import ctypes
    SendInput = ctypes.windll.user32.SendInput

    # C struct redefinitions
    PUL = ctypes.POINTER(ctypes.c_ulong)
    class KeyBdInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort),
                    ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]

    class HardwareInput(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong),
                    ("wParamL", ctypes.c_short),
                    ("wParamH", ctypes.c_ushort)]

    class MouseInput(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long),
                    ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong),
                    ("dwFlags", ctypes.c_ulong),
                    ("time",ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]

    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput),
                    ("mi", MouseInput),
                    ("hi", HardwareInput)]

    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong),
                    ("ii", Input_I)]

    def PressKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def ReleaseKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
else:
    def PressKey(hexKeyCode):
        pass
    def ReleaseKey(hexKeyCode):
        pass

# DirectInput Lookup table
DI_KEY_CODE_LOOKUP = {
    "DI_ESCAPE": 0x01,
    "DI_1": 0x02,
    "DI_2": 0x03,
    "DI_3": 0x04,
    "DI_4": 0x05,
    "DI_5": 0x06,
    "DI_6": 0x07,
    "DI_7": 0x08,
    "DI_8": 0x09,
    "DI_9": 0x0A,
    "DI_0": 0x0B,
    "DI_MINUS": 0x0C,  # - on main keyboard
    "DI_EQUALS": 0x0D,
    "DI_BACK": 0x0E,  # backspace
    "DI_TAB": 0x0F,
    "DI_Q": 0x10,
    "DI_W": 0x11,
    "DI_E": 0x12,
    "DI_R": 0x13,
    "DI_T": 0x14,
    "DI_Y": 0x15,
    "DI_U": 0x16,
    "DI_I": 0x17,
    "DI_O": 0x18,
    "DI_P": 0x19,
    "DI_LBRACKET": 0x1A,
    "DI_RBRACKET": 0x1B,
    "DI_RETURN": 0x1C,  # Enter on main keyboard
    "DI_LCONTROL": 0x1D,
    "DI_A": 0x1E,
    "DI_S": 0x1F,
    "DI_D": 0x20,
    "DI_F": 0x21,
    "DI_G": 0x22,
    "DI_H": 0x23,
    "DI_J": 0x24,
    "DI_K": 0x25,
    "DI_L": 0x26,
    "DI_SEMICOLON": 0x27,
    "DI_APOSTROPHE": 0x28,
    "DI_GRAVE": 0x29,  # accent grave
    "DI_LSHIFT": 0x2A,
    "DI_BACKSLASH": 0x2B,
    "DI_Z": 0x2C,
    "DI_X": 0x2D,
    "DI_C": 0x2E,
    "DI_V": 0x2F,
    "DI_B": 0x30,
    "DI_N": 0x31,
    "DI_M": 0x32,
    "DI_COMMA": 0x33,
    "DI_PERIOD": 0x34,  # . on main keyboard
    "DI_SLASH": 0x35,  # / on main keyboard
    "DI_RSHIFT": 0x36,
    "DI_MULTIPLY": 0x37,  # * on numeric keypad
    "DI_LMENU": 0x38,  # left Alt
    "DI_SPACE": 0x39,
    "DI_CAPITAL": 0x3A,
    "DI_F1":              0x3B,
    "DI_F2":              0x3C,
    "DI_F3":              0x3D,
    "DI_F4":              0x3E,
    "DI_F5":              0x3F,
    "DI_F6":              0x40,
    "DI_F7":              0x41,
    "DI_F8":              0x42,
    "DI_F9":              0x43,
    "DI_F10":             0x44,
    "DI_NUMLOCK":         0x45,
    "DI_SCROLL":          0x46,     # Scroll Lock
    "DI_NUMPAD7":         0x47,
    "DI_NUMPAD8":         0x48,
    "DI_NUMPAD9":         0x49,
    "DI_SUBTRACT":        0x4A,    # - on numeric keypad
    "DI_NUMPAD4":         0x4B,
    "DI_NUMPAD5":         0x4C,
    "DI_NUMPAD6":         0x4D,
    "DI_ADD":             0x4E,    # + on numeric keypad
    "DI_NUMPAD1":         0x4F,
    "DI_NUMPAD2":         0x50,
    "DI_NUMPAD3":         0x51,
    "DI_NUMPAD0":         0x52,
    "DI_DECIMAL":         0x53,    # . on numeric keypad
    "DI_F11":             0x57,
    "DI_F12":             0x58,

    "DI_F13":             0x64,    #                     (NEC PC98)
    "DI_F14":             0x65,    #                     (NEC PC98)
    "DI_F15":             0x66,    #                     (NEC PC98)

    "DI_KANA":            0x70,    # (Japanese keyboard)
    "DI_CONVERT":         0x79,    # (Japanese keyboard)
    "DI_NOCONVERT":       0x7B,    # (Japanese keyboard)
    "DI_YEN":             0x7D,    # (Japanese keyboard)
    "DI_NUMPADEQUALS":    0x8D,    # = on numeric keypad (NEC PC98)
    "DI_CIRCUMFLEX":      0x90,    # (Japanese keyboard)
    "DI_AT":              0x91,    #                     (NEC PC98)
    "DI_COLON":           0x92,    #                     (NEC PC98)
    "DI_UNDERLINE":       0x93,    #                     (NEC PC98)
    "DI_KANJI":           0x94,    # (Japanese keyboard)
    "DI_STOP":            0x95,    #                     (NEC PC98)
    "DI_AX":              0x96,    #                     (Japan AX)
    "DI_UNLABELED":       0x97,    #                        (J3100)
    "DI_NUMPADENTER":     0x9C,    # Enter on numeric keypad
    "DI_RCONTROL":        0x9D,
    "DI_NUMPADCOMMA":     0xB3,    # , on numeric keypad (NEC PC98)
    "DI_DIVIDE":          0xB5,    # / on numeric keypad
    "DI_SYSRQ":           0xB7,
    "DI_RMENU":           0xB8,    # right Alt
    "DI_HOME":            0xC7,    # Home on arrow keypad
    "DI_UP":              0xC8,    # UpArrow on arrow keypad
    "DI_PRIOR":           0xC9,    # PgUp on arrow keypad
    "DI_LEFT":            0xCB,    # LeftArrow on arrow keypad
    "DI_RIGHT":           0xCD,    # RightArrow on arrow keypad
    "DI_END":             0xCF,    # End on arrow keypad
    "DI_DOWN":            0xD0,    # DownArrow on arrow keypad
    "DI_NEXT":            0xD1,    # PgDn on arrow keypad
    "DI_INSERT":          0xD2,    # Insert on arrow keypad
    "DI_DELETE":          0xD3,    # Delete on arrow keypad
    "DI_LWIN":            0xDB,    # Left Windows key
    "DI_RWIN":            0xDC,    # Right Windows key
    "DI_APPS":            0xDD     # AppMenu key
}
