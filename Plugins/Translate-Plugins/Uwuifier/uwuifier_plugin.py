# ============================================================
# Uwuifier Plugin for Whispering Tiger
# V1.0.0
# Whispering Tiger: https://github.com/Sharrnah/whispering-ui
#
# This plugin transforms text into a playful "uwu" style.
# ============================================================
#

import json
import random
import re

import Plugins
import settings
import websocket

LANGUAGES = {
    "none": "No Translation",
    "uwu": "Uwuify",
}


class UwuifierPlugin(Plugins.Base):
    faces = [
        "owo", "uwu", "OwO", "UwU", ">w<", "^w^", "(・`ω´・)", "(uwu)", "(｡♥‿♥｡)", "(ᵘʷᵘ)", "(• o •)",
    ]
    actions = [
        "*blushes*", "*nuzzles*", "*pounces*", "*giggles*", "*wags tail*", "*snuggles*", "*squishes*",
    ]

    def init(self):
        # Plugin settings for intensity and optional effects
        self.init_plugin_settings(
            {
                # When to apply: always, or only if target language is 'uwu'
                "apply_when": {"type": "select", "value": "target_is_uwu", "values": ["always", "target_is_uwu"]},

                "intensity": {"type": "slider", "min": 0, "max": 3, "step": 1, "value": 2},
                "faces_enabled": False,
                "faces_chance": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "value": 0.30},
                "stutter_enabled": False,
                "stutter_chance": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "value": 0.10},
                "actions_enabled": False,
                "actions_chance": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "value": 0.10},
                "tildes_enabled": True,
                "tilde_chance": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "value": 0.20},
            },
            settings_groups={
                "General": [
                    "apply_when",
                    "intensity",
                    "faces_enabled", "faces_chance",
                    "stutter_enabled", "stutter_chance",
                    "actions_enabled", "actions_chance",
                    "tildes_enabled", "tilde_chance",
                ],
            },
        )

        if self.is_enabled(False):
            # Ensure built-in translator is not used when this plugin is active
            settings.SetOption("txt_translator", "")
            # Provide a language list to the UI
            websocket.BroadcastMessage(json.dumps({
                "type": "installed_languages",
                "data": self.return_languages(),
            }))

    def _replace_rl_with_w(self, text: str) -> str:
        text = re.sub(r"r", "w", text)
        text = re.sub(r"l", "w", text)
        text = re.sub(r"R", "W", text)
        text = re.sub(r"L", "W", text)
        return text

    def _nyeh(self, text: str) -> str:
        # n + vowel -> ny + vowel (simple, preserves case for first letter)
        text = re.sub(r"n([aeiou])", r"ny\1", text)
        text = re.sub(r"N([aeiou])", r"Ny\1", text)
        text = re.sub(r"N([AEIOU])", r"NY\1", text)
        return text

    def _ove_to_uv(self, text: str) -> str:
        # love -> wuv (after rl->w, this helps words like "love")
        def repl(m):
            s = m.group(0)
            # keep case of first char
            return "uv" if s.islower() else "Uv"
        return re.sub(r"(?i)ove", repl, text)

    def _th_to_d(self, text: str) -> str:
        text = re.sub(r"th", "d", text)
        text = re.sub(r"Th", "D", text)
        text = re.sub(r"TH", "D", text)
        text = re.sub(r"tH", "d", text)
        return text

    def _stutter_word(self, word: str) -> str:
        if len(word) > 2 and word[0].isalpha():
            # preserve casing for first char
            return f"{word[0]}-{word}"
        return word

    def _insert_faces(self, text: str, chance: float) -> str:
        def add_face(m):
            if random.random() < chance:
                return m.group(0) + " " + random.choice(self.faces)
            return m.group(0)
        # after ! or ? optionally add a face
        text = re.sub(r"([!?]+)", add_face, text)
        return text

    def _maybe_add_action(self, text: str, chance: float) -> str:
        if random.random() < chance:
            return text.rstrip() + " " + random.choice(self.actions)
        return text

    def _maybe_add_tildes(self, text: str, chance: float) -> str:
        # Occasionally replace a comma or end-of-sentence space with a tilde-y vibe
        def tilde_space(m):
            return " ~ " if random.random() < chance else m.group(0)
        text = re.sub(r"\s,\s", tilde_space, text)
        text = re.sub(r"\s\.\s", tilde_space, text)
        return text

    def _protect_abbreviations(self, text: str):
        """Replace ALL-CAPS words (length>=2) with placeholders."""
        mapping = {}

        def repl(m):
            key = f"[[{len(mapping)}]]"  # placeholder, safe from rl/ny/ove/th transforms
            mapping[key] = m.group(0)
            return key

        protected_text = re.sub(r"\b[A-Z0-9]{2,}\b", repl, text)
        return protected_text, mapping

    def _restore_abbreviations(self, text: str, mapping: dict) -> str:
        # Replace placeholders back; iterating keys is safe because markers are distinct
        for key, value in mapping.items():
            text = text.replace(key, value)
        return text

    def uwufy_text(self, text: str) -> str:
        """
        Transform input text to a playful "uwu" style. No external dependencies.
        Intensity levels:
        0 = only r/l -> w
        1 = + n->ny before vowels
        2 = + ove->uv, faces/tildes if enabled
        3 = + th->d
        """
        if not isinstance(text, str) or text.strip() == "":
            return text

        # Protect abbreviations like 'VR' from being modified
        text, abbrev_map = self._protect_abbreviations(text)

        intensity = int(self.get_plugin_setting("intensity"))
        faces_enabled = bool(self.get_plugin_setting("faces_enabled"))
        faces_chance = float(self.get_plugin_setting("faces_chance"))
        stutter_enabled = bool(self.get_plugin_setting("stutter_enabled"))
        stutter_chance = float(self.get_plugin_setting("stutter_chance"))
        actions_enabled = bool(self.get_plugin_setting("actions_enabled"))
        actions_chance = float(self.get_plugin_setting("actions_chance"))
        tildes_enabled = bool(self.get_plugin_setting("tildes_enabled"))
        tilde_chance = float(self.get_plugin_setting("tilde_chance"))

        # Word-level stutter pass (before character transforms)
        if stutter_enabled and stutter_chance > 0:
            words = text.split(" ")
            for i, w in enumerate(words):
                if random.random() < stutter_chance:
                    words[i] = self._stutter_word(w)
            text = " ".join(words)

        # Character-level transformations
        text = self._replace_rl_with_w(text)
        if intensity >= 1:
            text = self._nyeh(text)
        if intensity >= 2:
            text = self._ove_to_uv(text)
        if intensity >= 3:
            text = self._th_to_d(text)

        # Decorations
        if faces_enabled and faces_chance > 0:
            text = self._insert_faces(text, faces_chance)
        if tildes_enabled and tilde_chance > 0:
            text = self._maybe_add_tildes(text, tilde_chance)
        if actions_enabled and actions_chance > 0:
            text = self._maybe_add_action(text, actions_chance)

        # Restore protected abbreviations
        text = self._restore_abbreviations(text, abbrev_map)

        return text

    def text_translate(self, text, from_code, to_code) -> tuple:
        """
        Rewrite text to uwu style. Applies either always or only when target is 'uwu', depending on setting.
        """
        if not self.is_enabled(False):
            return text, from_code, to_code

        apply_when = self.get_plugin_setting("apply_when")
        should_apply = apply_when == "always" or (apply_when == "target_is_uwu" and str(to_code).lower() == "uwu")
        if not should_apply:
            return text, from_code, to_code

        translated_text = self.uwufy_text(text)
        detected_language = from_code or ""
        return translated_text, (detected_language.lower() if isinstance(detected_language, str) else detected_language), to_code

    # --- Language plumbing for UI ---
    def return_languages(self):
        return tuple([{"code": code, "name": language} for code, language in LANGUAGES.items()])

    def on_plugin_get_languages_call(self, data_obj):
        if self.is_enabled(False):
            data_obj['languages'] = self.return_languages()
            return data_obj
        return None

    def on_enable(self):
        self.init()
        # nothing else special on enable
        pass
