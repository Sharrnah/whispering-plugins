# Plugin Creation

Plugins are a way to extend the functionality of Whispering Tiger.

There mignt still be changes to the plugin system in the future.

## How to use

Plugins are loaded from the `Plugins` directory in the root of the project. The directory is scanned for `.py` files and each file is loaded as a plugin.

## How to write

Plugins are written as Python classes. The class must inherit from `Plugins.Base` and implement at least the `init` method.

At the very top of the file, you should add a comment with a short description and most importantly, a version line, so the version can be compared inside the UI application.
example:
```python
# ============================================================
# This is the plugin xyz for Whispering Tiger
# Version: 1.0.0
# some more information about the plugin
# ============================================================
```
The format of the version line can start with `Version: `, `Version `, `V`, `V: ` followed by `<major>.<minor>.<patch>`

| Function                                                                            | Optionality | Description                                                                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `init(self)`                                                                        | Required    | is called at the initialization of whispering tiger, right after the settings file is loaded.                                                                                                                                                                                                                              |
| `__plugin_init__(self)`                                                             | Optional    | is called at the initialization of the Plugins, before settings file is loaded. (Only use if really needed.)                                                                                                                                                                                                               |
| `on_enable(self)`                                                                   | Optional    | is called when the plugin is enabled.                                                                                                                                                                                                                                                                                      |
| `on_disable(self)`                                                                  | Optional    | is called when the plugin is disabled.                                                                                                                                                                                                                                                                                     |
| `timer(self)`                                                                       | Optional    | is called every x seconds (defined in `plugin_timer`) and is paused for x seconds (defined in `plugin_timer_timeout`) when the Speech-to-Text engine returned a result.<br>_This can be used for a regular output that is stopped occasionally when a more important transcription is supposed to be displayed._           |
| `stt(self, text, result_obj)`                                                       | Optional    | is called when the Speech-to-Text engine returns a result.                                                                                                                                                                                                                                                                 |
| `stt_intermediate(self, text, result_obj)`                                          | Optional    | is called when a live transcription result is available. Make sure to use the `stt` function for final results.                                                                                                                                                                                                            |
| `stt_processing(self, audio_data, sample_rate, final_audio) -> dict (data_obj)`     | Optional    | is called when audio is recorded and no local Speech-to-Text model is loaded (useful for Speech-to-Text Plugins).<br>Needs to return a result object in the format:<br>`{'text': transcribed_text, 'type': "transcript", 'language': source_language}`<br>audio_data = raw 16 bit float mono audio<br>sample_rate = 16000. |
| `tts(self, text, device_index, websocket_connection=None, download=False, path='')` | Optional    | is called when the TTS engine is about to play a result, except when called by the sst engine.<br>_if you want to play a sound when the Speech-to-Text engine returns a result, you should do it in the `stt` method as well. Path is set if the TTS request was to save the TTS result._                                  |
| `get_last_generation(self) -> tuple (wav_bytes, sample_rate)`                       | Optional    | is called when the user requests the last generated TTS result.                                                                                                                                                                                                                                                            |
| `sts(self, wavefiledata, sample_rate)`                                              | Optional    | is called when a recording is finished (which is sent to the Speech-to-Text model). This function gets the audio recording to be processed by the plugin.                                                                                                                                                                  |
| `text_translate(self, text, from_code, to_code) -> tuple (txt, from_lang, to_lang)` | Optional    | is called when a translation is requested and no included translator is available.<br>_Must return a tuple of translation_text, from_lang_code, to_lang_code._                                                                                                                                                             |
| `on_{event_name}_call(self, data_obj) -> dict (data_obj)`                           | Optional    | is called when a custom plugin event is called via `Plugins.plugin_custom_event_call(event_name, data_obj)`. See [Custom Plugin events](#Custom-Plugin-events) for more info.                                                                                                                                              |

## Helper methods

The `Base` class provides some helper methods to make it easier to write plugins.

`init_plugin_settings(self, settings, settings_groups=None)` - Prepare all possible plugin settings and their default values. This method must be called in the `init` method of the plugin. The `settings` parameter is a dictionary with the settings and their default values. The `settings_groups` parameter is an optional dictionary with the settings groups and the settings that belong to that group. The settings groups are used in the settings window to group the settings. If the `settings_groups` parameter is not provided no groups are displayed in the UI.

using a 2 dimensional array per group will result in splitting the settings widgets into columns.

**IMPORTANT: Settings that are not initialized with `init_plugin_settings` are deleted when calling `init_plugin_settings`, so make sure to define every setting your Plugin needs.**

`get_plugin_setting(self, setting, default=None)` - Get a plugin setting from the settings file. If the setting is not yet in the settings file, the default value is used. (if default is not set, the default from _init_plugin_settings()_ is used)

`set_plugin_setting(self, setting, value)` - Set a plugin setting in the settings file.

__Note:__ _When using the *_plugin_setting methods, the settings are saved with the class name as the section name. So if you have a plugin called `ExamplePlugin`, the settings will be saved in the `ExamplePlugin` section._

`is_enabled(self, default=False)` - Check if the plugin is enabled. If the plugin is not yet in the settings file, the default value is used. So by default, plugins will be disabled. Use this around your main functionality to allow enabling/disabling of your plugin functionality even at runtime.


## Use specific Widgets in plugin settings

To use specific widgets in plugin settings, you can add specific structs to the init_plugin_settings method.

The following structs are available:
- `{"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.7}` - A slider
- `{"type": "button", "label": "Batch Generate", "style": "primary"}` - A button (style can be "primary" or "default")
- `{"type": "select", "label": "Label", "value": "default value", "options": ["default value", "option2", "option3"]}` - A select box
- `{"type": "textarea", "rows": 5, "value": ""}` - A textarea
- `{"type": "textfield", "password": false, "value": ""}` - A textfield (password field if "password" is true)
- `{"type": "hyperlink", "label": "hyperlink", "value": "https://github.com/Sharrnah/whispering-ui"}`
- `{"type": "label", "label": "Some infotext in a label.", "style": "center"}` - A label (style can be "left", "right" or "center")
- `{"type": "file_open", "accept": ".wav,.mp3", "value": "bark_clone_voice/clone_voice.wav"}` - A file open dialog (accept can be any file extension or a comma separated list of file extensions)
- `{"type": "file_save", "accept": ".npz", "value": "last_prompt.npz"}` - A file save dialog (accept can be any file extension or a comma separated list of file extensions)
- `{"type": "folder_open", "accept": "", "value": ""}` - A folder open dialog
- `{"type": "dir_open", "accept": "", "value": ""}` - Alias for a folder open dialog
- `{"type": "select_audio", "device_api": "|wasapi|mme|directsound|all", "device_type": "input|output", "value": ""}` - List of audio devices. Only listing input / output devices if device_type = "input" / device_type = "output". "device_api" can be empty (using main app api), 'wasapi', 'mme' or 'directsound' for a specific audio api, or 'all' for all APIs. `get_plugin_setting` on a `select_audio` will return a valid pyAudio Audio Device ID.
- `{"type": "select_textvalue", "value": "default value", "options": [["The Default Value Text", "default value"], ["The Option 2 text", "option2"], ["option 3 text", "option3"]]}` - A select box with shown text and value seperately. Each option is an array where the key is the displayed text to the user and the value is the value returned by `get_plugin_setting`.
- `{"type": "select_completion", "value": "English", "options": [["English", "en"], ["French", "fr"], ["German", "de"]]}` - A Input field with autocompletion similar to the language fields in the main application. Each option is an array where the key is the displayed text to the user and the value is the value returned by `get_plugin_setting`.

## Custom Plugin events
You can use event calls in plugins using `Plugins.plugin_custom_event_call(event_name, data_obj)`. This will give the result from the first Plugin with that event function.

To get the results of all Plugins with that event function, use `Plugins.plugin_custom_event_call_all(event_name, data_obj)` which will return a list of all results you can iterate over.

The function names have the form of `on_{event_name}_call`.

`event_name` should be unique and self explaining.

The function needs to return `None` if something failed or should be skipped,
or the `data_obj` again with your necessary changes to the object.

**Note:** The order in which plugins are loaded and their event functions are called is not defined.

As an example the call from the Silero or F5 TTS:
```py
plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': audio, 'sample_rate': self.sample_rate})
if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
    audio = plugin_audio['audio']
```

The called function in the Plugin looks like this:
```py
def on_plugin_tts_after_audio_call(self, data_obj):
    if self.is_enabled(False) and self.get_plugin_setting("voice_change_source") == CONSTANTS["PLUGIN_TTS"]:
        audio = data_obj['audio']
        
        # doing stuff ...
        
        # set modified audio back to data_obj
        data_obj['audio'] = audio
        return data_obj
    return None
```

Before calling Events from other Plugins, make sure all Plugins are already loaded. (Should not be called in `__init__`).

Make sure to check if the Event should be callable. (Is Plugin enabled? Are the plugin settings configured properly? ...). Otherwise return `None`.

### List of core plugin events

| Function Name                                                  | Description                                                                                                                                                                                                                                                                                                                                                                                                               |
|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| on_plugin_tts_after_audio_call(data_obj)                       | Called after the Included TTS generated audio. Expects the `audio` and `sample_rate` key in the `data_obj` <br> (audio as Bytes or Pytorch Tensor).                                                                                                                                                                                                                                                                                           |
| on_audio_processor_stt_{audio_processor_caller}_call(data_obj) | Called after the stt and stt_intermediate plugin methods are called, <br>with the name of the `audio_processor_caller` setting _`settings.SetOption("audio_processor_caller", "custom_name")`_.<br>`data_obj` will be `{"text": predicted_text, "result_obj": result_obj, "final_audio": final_audio}` (used by the [Secondary Profile plugin](/Plugins/Other-Plugins/Secondary-Profile/secondary_profile_plugin.py))<br> |
| on_plugin_get_languages_call(data_obj)                         | Called to fetch available text translation languages for Text Translation Plugins. should return the `data_obj` with a dict in the form `data_obj['languages'] = tuple([{"code": code, "name": language}])`. Return `None` if the Plugin is disabled.<br>                                                                                                                                                                 |
| on_plugin_llm_function_registration_call(data_obj)                         | Called to register LLM function call methods (Currently used by Phi-4 MM Model). See [Phi4 function-call weather plugin](/Plugins/Other-Plugins/LLM-function-calling/phi4_function_call_weather_plugin.py)<br>                                                                                                                                                                 |

## Example plugin
```python
# ============================================================
# This is the example plugin for Whispering Tiger
# Version: 1.0.0
# some more information about the plugin
# ============================================================
import Plugins
import settings
import VRC_OSCLib

class ExamplePlugin(Plugins.Base):
    last_generation = {"audio": None, "sample_rate": None}
    
    def init(self):
        # prepare all possible plugin settings and their default values
        self.init_plugin_settings(
            {
                "hello_world": "default value",
                "hello_world2": "foo bar",
                "osc_auto_processing_enabled": False,
                "tts_answer": False,
                "homepage_link": {"label": "Whispering Tiger Link", "value": "https://whispering-tiger.github.io/", "type": "hyperlink"},

                "more_settings_a": "default value",
                "more_settings_b": "default value\nmultiline",
                "more_settings_c": 0.15,
                "more_settings_d": 60,
            },
            settings_groups={
                "General": ["osc_auto_processing_enabled", "tts_answer", "hello_world", "hello_world2", "homepage_link"],
                "Second Group": ["more_settings_a", "more_settings_b", "more_settings_c", "more_settings_d"],
                "2 Columns": [
                    ["column1_setting_a", "column1_setting_b"], # Column 1
                    ["column2_setting_a", "column2_setting_b"]  # Column 2
                ],
            }
        )
        
        if self.is_enabled():
            print(self.__class__.__name__ + " is enabled")

            # disable OSC processing so the Plugin can take it over:
            settings.SetOption("osc_auto_processing_enabled", False)
            # disable TTS so the Plugin can take it over:
            settings.SetOption("tts_answer", False)

            # disable websocket final messages processing so the Plugin can take it over:
            # this is really only needed if you want to use the websocket to send your own messages.
            # for the Websocket clients to understand the messages, you must follow the format. (see the LLM Plugin for a good example)
            ## settings.SetOption("websocket_final_messages", False)
        else:
            print(self.__class__.__name__ + " is disabled")

    ## OPTIONAL. called every x seconds (defined in plugin_timer)
    def timer(self):
        # get the settings from the global app settings
        osc_ip = settings.GetOption("osc_ip")
        osc_address = settings.GetOption("osc_address")
        osc_port = settings.GetOption("osc_port")

        # get plugin settings
        hello_world = self.get_plugin_setting("hello_world", "default foo bar")
        hello_world2 = self.get_plugin_setting("hello_world2") # if no default is defined, default is taken from init_plugin_settings
        print(hello_world2)

        if self.is_enabled():
            VRC_OSCLib.Chat(hello_world, True, False, osc_address, IP=osc_ip, PORT=osc_port,
                            convert_ascii=False)
        pass

    ## OPTIONAL. called when the STT engine returns a result
    def stt(self, text, result_obj):
        if self.is_enabled():
            print("Plugin Example")
            print(result_obj['language'])
        return

    ## OPTIONAL. only called when the STT engine returns an intermediate live result
    def stt_intermediate(self, text, result_obj):
        if self.is_enabled():
            print("Plugin Example")
            print(result_obj['language'])
        return
    
    ## OPTIONAL. only called when audio is recorded and no STT model is loaded. (Can be used for Speech-to-Text Plugins)
    # audio_data is raw audio bytes in 16 bit float mono audio and sample_rate of 16000.
    def stt_processing(self, audio_data, sample_rate, final_audio) -> dict|None:
        if self.is_enabled():
            if final_audio:
                print("Plugin Example, process the audio_data to get text")
                return {
                    'text': "transcribed_text",  # the transcribed text
                    'type': "transcript",  # needs always be "transcript"!
                    'language': "source_language"  # the recognized spoken language
                }
        return None

    ## OPTIONAL. called when the "send TTS" function is called
    def tts(self, text, device_index, websocket_connection=None, download=False, path=''):
        #wav = self.generate_tts(text.strip())  # generate the TTS audio
        if wav is not None:
            if download:
                if path is not None and path != '':
                    # write wav_data to file in path
                    with open(path, "wb") as f:
                        f.write(wav)
                    websocket.BroadcastMessage(json.dumps({"type": "info",
                                                            "data": "File saved to: " + path}))
            # save last generation in memory
            self.last_generation = {"audio": wav, "sample_rate": sample_rate}
        return

    # OPTIONAL - called to get last TTS result audio
    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]
    
    ## OPTIONAL - called when audio is finished recording and the audio is sent to the STT model
    def sts(self, wavefiledata, sample_rate):
        return

    ## OPTIONAL - called when translation is requested and no other translator is selected. must return a tuple consisting of text, from_code, to_code.
    def text_translate(self, text, from_code, to_code) -> tuple:
        return text, from_code, to_code
    
    ## OPTIONAL - called when a websocket message is received.
    ## formats are: (where 'ExamplePlugin' is the plugin class name)
    ## {"name": "ExamplePlugin", "type": "plugin_button_press", "value": "button_name"}
    ## {"name": "ExamplePlugin", "type": "plugin_custom_event", "value": []}
    def on_event_received(self, message, websocket_connection=None):
        if "type" not in message:
            return
        if message["type"] == "plugin_button_press":
            if message["value"] == "button_name":
                print("button pressed")
        if message["type"] == "plugin_custom_event":
            if message["value"] == "other_event_name":
                print("custom event received")
        pass
    
    ## OPTIONAL
    def on_enable(self):
        pass
    
    ## OPTIONAL
    def on_disable(self):
        pass

    ## OPTIONAL - custom event call function.
    # def on_{event_name}_call(self, data_obj):
    #     if self.is_enabled(False):
    #         return data_obj
    # return None
```
