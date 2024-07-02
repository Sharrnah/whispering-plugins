# ============================================================
# Translates Text using DeepL API - Whispering Tiger Plugin
# Version 1.0.7
# See https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import json
import requests

import Plugins
import settings

import websocket

DEEPL_ENDPOINTS = {
    "Free": "https://api-free.deepl.com/v2/translate",
    "Pro": "https://api.deepl.com/v2/translate",
    "DeepLX": ""
}

LANGUAGES = {
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
    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "api": {"type": "select", "value": 'Free',
                        "values": ['Free', 'Pro', 'DeepLX']},
                "formality": {"type": "select", "value": 'default',
                              "values": ['default', 'prefer_more', 'prefer_less']},
                "auth_key": {"type": "textfield", "value": "", "password": True},
                "deeplx_endpoint": "",
                "deeplx_info_link": {"label": "Open DeepLX GitHub", "value": "https://github.com/OwO-Network/DeepLX", "type": "hyperlink"},
            },
            settings_groups={
                "General": ["api", "formality", "auth_key"],
                "DeepLX": ["deeplx_endpoint", "deeplx_info_link"],
            }
        )

        if self.is_enabled(False):
            # disable txt-translator AI model if plugin is enabled
            settings.SetOption("txt_translator", '')

    def _translate_text_api(self, text, source_lang, target_lang, auth_key):
        url = DEEPL_ENDPOINTS[self.get_plugin_setting("api")]
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

        if source_lang is not None and source_lang not in ['auto', '']:
            data['source_lang'] = source_lang

        if formality is not None and formality not in ['default', '']:
            data['formality'] = formality

        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            websocket.BroadcastMessage(json.dumps({"type": "error", "data": "Error translating text ("+str(response.status_code)+"): " + response.text}))
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

    def on_enable(self):
        self.init()
        if self.is_enabled(False):
            websocket.BroadcastMessage(json.dumps({"type": "installed_languages", "data": self.return_languages()}))
        pass
