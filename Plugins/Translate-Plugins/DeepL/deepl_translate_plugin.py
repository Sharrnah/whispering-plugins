# ============================================================
# Translates Text using DeepL API - Whispering Tiger Plugin
# Version 1.0.9
# See https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import json
import requests

import Plugins
import settings

import websocket
import time
import random

DEEPL_ENDPOINTS = {
    "Free": "https://api-free.deepl.com/v2/",
    "Pro": "https://api.deepl.com/v2/",
    "DeepLX": ""
}

LANGUAGES = {
    "AR": "Arabic",  # target language only
    "BG": "Bulgarian",
    "CS": "Czech",
    "DA": "Danish",
    "DE": "German",
    "EL": "Greek",
    "EN": "English",
    "EN-GB": "English (British)",  # target language only
    "EN-US": "English (American)",  # target language only
    "ES": "Spanish",
    "ET": "Estonian",
    "FI": "Finnish",
    "FR": "French",
    "HU": "Hungarian",
    "ID": "Indonesian",
    "IT": "Italian",
    "JA": "Japanese",
    "KO": "Korean",
    "LT": "Lithuanian",
    "LV": "Latvian",
    "NB": "Norwegian (Bokmål)",
    "NL": "Dutch",
    "PL": "Polish",
    "PT": "Portuguese",
    "PT-BR": "Portuguese (Brazilian)",  # target language only
    "PT-PT": "Portuguese (excluding Brazilian)",  # target language only
    "RO": "Romanian",
    "RU": "Russian",
    "SK": "Slovak",
    "SL": "Slovenian",
    "SV": "Swedish",
    "TR": "Turkish",
    "UK": "Ukrainian",
    "ZH": "Chinese",
}


class DeepLPlugin(Plugins.Base):
    max_retries = 5
    base_delay = 1

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "api": {"type": "select", "value": 'Free',
                        "values": ['Free', 'Pro', 'DeepLX']},
                "formality": {"type": "select", "value": 'default',
                              "values": ['default', 'prefer_more', 'prefer_less']},
                "model_type": {"type": "select", "value": 'prefer_quality_optimized',
                              "values": ['prefer_quality_optimized', 'quality_optimized', 'latency_optimized']},
                "auth_key": {"type": "textfield", "value": "", "password": True},
                "quota_btn": {"label": "Check Usage Quota", "type": "button", "style": "default"},
                "deeplx_endpoint": "",
                "deeplx_info_link": {"label": "Open DeepLX GitHub", "value": "https://github.com/OwO-Network/DeepLX", "type": "hyperlink"},
            },
            settings_groups={
                "General": ["api", "formality", "auth_key", "model_type", "quota_btn"],
                "DeepLX": ["deeplx_endpoint", "deeplx_info_link"],
            }
        )

        if self.is_enabled(False):
            # disable txt-translator AI model if plugin is enabled
            settings.SetOption("txt_translator", '')
            websocket.BroadcastMessage(json.dumps({"type": "installed_languages", "data": self.return_languages()}))

    def _translate_text_api(self, text, source_lang, target_lang, auth_key):
        url = DEEPL_ENDPOINTS[self.get_plugin_setting("api")] + "translate"
        formality = self.get_plugin_setting("formality")
        headers = {
            'Authorization': f'DeepL-Auth-Key {auth_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'text': [text],
            'target_lang': target_lang
        }

        # special case for DeepLX endpoint
        if self.get_plugin_setting("api") == "DeepLX":
            url = self.get_plugin_setting("deeplx_endpoint")
            headers['Authorization'] = f'Bearer {auth_key}'
            data['text'] = text

        # Check and modify source_lang if it contains a "-"
        if source_lang is not None and '-' in source_lang:
            source_lang = source_lang.split('-')[0]

        if source_lang is not None and source_lang.lower() not in ['auto', '']:
            data['source_lang'] = source_lang

        if formality is not None and formality not in ['default', '']:
            data['formality'] = formality

        # model_type currently only supported by Pro API. (see https://developers.deepl.com/docs/api-reference/translate)
        if self.get_plugin_setting("api") == "Pro":
            data['model_type'] = self.get_plugin_setting("model_type")

        response = None

        for retry in range(self.max_retries):
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                break
            elif response.status_code == 429:
                if retry < self.max_retries - 1:
                    delay = (self.base_delay * 2 ** retry) + (random.randint(0, 1000) / 1000.0)
                    websocket.BroadcastMessage(json.dumps({"type": "warning", "data": f"Rate limit reached. Retrying in {delay:.2f} seconds..."}))
                    time.sleep(delay)
                else:
                    websocket.BroadcastMessage(json.dumps({"type": "error", "data": f"Max retries reached. Unable to translate text after {self.max_retries} attempts."}))
                    return "", ""
            else:
                websocket.BroadcastMessage(json.dumps({"type": "error", "data": f"Error translating text ({response.status_code}): {response.text}"}))
                return "", ""

        if response is None or response.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": "Unexpected error occurred during translation."}))
            return "", ""

        response_json = response.json()

        # special case for DeepLX endpoint
        if self.get_plugin_setting("api") == "DeepLX":
            translated_text = response_json['data']
            detected_language = response_json['source_lang']
            return translated_text, detected_language

        # Extracting the translated text
        translated_text = response_json['translations'][0]['text']
        detected_language = response_json['translations'][0]['detected_source_language']
        return translated_text, detected_language

    def check_quota(self):
        auth_key = self.get_plugin_setting("auth_key")
        url = DEEPL_ENDPOINTS[self.get_plugin_setting("api")] + "usage"
        headers = {
            'Authorization': f'DeepL-Auth-Key {auth_key}',
            'Content-Type': 'application/json'
        }
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": f"Error checking quota ({response.status_code}): {response.text}"}))
            return
        response_json = response.json()
        character_count = response_json['character_count']
        character_limit = response_json['character_limit']
        characters_remaining = character_limit - character_count

        # make the numbers more readable
        character_count = "{:,}".format(character_count)
        character_limit = "{:,}".format(character_limit)
        characters_remaining = "{:,}".format(characters_remaining)

        websocket.BroadcastMessage(json.dumps({"type": "info", "data": f"Characters used: {character_count} / {character_limit}\nRemaining: {characters_remaining}"}))

    def text_translate(self, text, from_code, to_code) -> tuple:
        """
        on text_translate event, translates text using DeepL API.
        """
        if self.is_enabled(False):
            auth_key = self.get_plugin_setting("auth_key")
            translated_text, detected_language = self._translate_text_api(
                text=text, source_lang=from_code, target_lang=to_code, auth_key=auth_key
            )
            return translated_text, detected_language.lower(), to_code

    def return_languages(self):
        return tuple([{"code": code, "name": language} for code, language in LANGUAGES.items()])

    def on_plugin_get_languages_call(self, data_obj):
        if self.is_enabled(False):
            data_obj['languages'] = self.return_languages()
            return data_obj

        return None

    def on_event_received(self, message, websocket_connection=None):
        if self.is_enabled(False):
            if "type" not in message:
                return
            if message["type"] == "plugin_button_press":
                if message["value"] == "quota_btn":
                    self.check_quota()

    def on_enable(self):
        self.init()
        pass
