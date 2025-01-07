# <img src=https://github.com/Sharrnah/whispering-ui/blob/main/app-icon.png width=90> Plugins for Whispering Tiger

Current list of plugins that are available for [Whispering Tiger](https://github.com/Sharrnah/whispering-ui).

_If you have created a plugin, please add it to this list using a pull request or let me know and I will add it._

_See [plugin-creation.md](Documentation/plugin-creation.md) for more information on how to create plugins yourself._

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/sharrnah)

### Installation of Plugins
#### Using Whispering Tiger UI Application _(recommended)_:
Go to the `Plugins` Tab and click on the `Download / Update Plugins` Button.

Select the Plugin you want to install from the list and press `Install`.

_(The Button might show `ReInstall` or `Update` depending on if the Plugin is already installed and is the current version.)_

#### Manual Installation:
Download the `.py` file and copy the plugin file into the `Plugins` directory in the root of the Whispering Tiger folder.

## List of Plugins

| Title                                                                                                                                                                                                | Preview                                                                                                                                                                                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Author     |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| [**Keyboard Typing**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/Other-Plugins/Keyboard-Typing/keyboard_typing_plugin.py)                                                      | <img src=Previews/Keyboard-Typing.gif width=250>                                                                                                                                            | Type Text without your keyboard but instead with your Voice. <br> supports additional _customizable_ commands like: <br> - start typing<br>- stop typing<br>- new line                                                                                                                                                                                                                                                                                                                                    | Sharrnah   |
| [**Subtitle Display**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/Other-Plugins/Subtitles-Display/subtitles_display_plugin.py)                                                 | <img src=Previews/Subtitle-Display.gif width=250>                                                                                                                                           | Display Subtitles everywhere on your Desktop                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Sharrnah   |
| [**Subtitles Export**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/Export-Plugins/Subtitles-Export/subtitles_export_plugin.py)                                                  | <img src=Previews/Subtitles-Export.png width=250>                                                                                                                                           | Generate Subtitle files for Audio or Video files.<br>_Can export as .VTT, .SRT or .SBV_                                                                                                                                                                                                                                                                                                                                                                                                                   | Sharrnah   |
| [**Secondary Profile**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/Other-Plugins/Secondary-Profile/secondary_profile_plugin.py)                                                | <img src=Previews/Secondary-Profile.png width=250>                                                                                                                                          | Load a secondary Profile at the same time, supporting a second recording and playback device selection. Does not load AI models a second time into memory.                                                                                                                                                                                                                                                                                                                                                | Sharrnah   |
| [**Voicevox Text 2 Speech**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/TTS-Plugins/Voicevox/voicevox_tts_plugin.py)                                                           | <video src='https://github.com/Sharrnah/whispering-plugins/assets/55756126/f56183a6-5264-4533-8bf5-2c32eca50f0a' width=100></video>                                                         | Japanese Text 2 Speech.</br>change **speaker** of selected model in Text-to-Speech tab.</br>**acceleration_mode:** can be "CPU" or "CUDA" </br><sub>thx to https://voicevox.hiroshiba.jp/ </sub>                                                                                                                                                                                                                                                                                                          | Sharrnah   |
| [**TALQu3PRO Text 2 Speech**](https://github.com/rokujyushi/WT2TALQu/blob/main/TALQu3PRO_TTS.py)                                                                                                     |                                                                                                                                                                                             | Japanese Text 2 Speech.</br>Set the path to TALQuClient in General.</br><sub>thx to https://haruqa.github.io/TALQu/ </sub>                                                                                                                                                                                                                                                                                                                                                                                | Rokujyushi |
| [**Bark Text 2 Speech**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/TTS-Plugins/Bark/tts_bark_plugin.py)                                                                       | <video src='https://github.com/Sharrnah/whispering-plugins/assets/55756126/fafbd5cc-6bb6-4c0b-8b6a-00068bf8157c' width=100></video>                                                         | Multilingual Text 2 Speech</br>change **history_prompt:** to one of the voices you can find here: [Bark Speaker Library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683) </br>**prompt_wrap** Setting can be used for more prompt features like:</br>- singing ("`♪ ## ♪`")</br>- bias towards male or female ("`[MAN] ##`" or "`[WOMAN] ##`")</br>- more infos in their [Readme](https://github.com/suno-ai/bark#-usage-in-python).</br><sub>thx to https://github.com/suno-ai/bark </sub> | Sharrnah   |
| [**Coqui Text 2 Speech**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/TTS-Plugins/Coqui/coqui_tts_plugin.py)                                                                    | _example generated with `tts_models/en/vctk/vits`_</br> <video src='https://github.com/Sharrnah/whispering-plugins/assets/55756126/e180ce5f-809d-448a-80df-ab84aec15c1b' width=100></video> | Multilingual Text 2 Speech</br> Supports many different TTS Models, including:</br>- Bark</br>- Tortoise v2</br>- VITS</br>- fairseq VITS (with ~1100 languages)</br>- and many more.</br>In addition it features a one-shot Voice Conversion model **FreeVC** which can be used with Text 2 Speech or Speech 2 Speech.</br><sup>_(Plugin uses a locally running Coqui Server)_</sup></br><sub>thx to https://github.com/coqui-ai/TTS/ </sub>                                                             | Sharrnah   |
| [**ElevenLabs Text 2 Speech**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/TTS-Plugins/Elevenlabs/elevenlabs_tts_plugin.py)                                                     | <video src='https://github.com/Sharrnah/whispering-plugins/assets/55756126/7cfef39b-a12d-4c64-9774-4df00a0ab7d5' width=100></video>                                                         | Multilingual Text 2 Speech (**API**).</br>Set **api_key:** to your API key.</br>change **voice:** to one of voices and **voice_index:** to the index of the voice. _(other than 0 if more voices with same name exist)_</br>**stt_*:** Settings can limit the generation to prevent accidental use up of available chars on your account.</br><sub>thx to https://elevenlabs.io/ </sub>                                                                                                                   | Sharrnah   |
| [**ChatTTS Text 2 Speech**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/TTS-Plugins/ChatTTS/tts_chattts_plugin.py)                                                              | <video src='https://github.com/Sharrnah/whispering-plugins/assets/55756126/f9866cda-0bf0-48a3-afe9-004bca1dd2c2' width=100></video>                                                         | (currently) English and Chinese Only Text 2 Speech</br><sub>thx to https://github.com/2noise/ChatTTS </sub>                                                                                                                                                                                                                                                                                                                                                                                               | Sharrnah   |
| [**Mars5 Text 2 Speech**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/TTS-Plugins/Mars5/tts_mars5_plugin.py)                                                                    | <video src='https://github.com/Sharrnah/whispering-plugins/assets/55756126/ed2ef592-adb6-4eff-a471-a05745be8596' width=100></video>                                                         | (currently) English Only Text 2 Speech</br><sub>thx to https://www.camb.ai/ </sub>                                                                                                                                                                                                                                                                                                                                                                                                                        | Sharrnah   |
| [**DeepL Text Translation**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/Translate-Plugins/DeepL/deepl_translate_plugin.py)                                                     | <img src=Previews/DeepL.png width=100>                                                                                                                                                      | DeepL Text Translation (**API**).</br>Set **auth_key:** to your Authentication key.</br></br>Be careful using it with realtime mode, as it might use up your characters fast.</br><sub>thx to https://www.deepl.com/ </sub>                                                                                                                                                                                                                                                                               | Sharrnah   |
| [**OpenAI API**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/Other-Plugins/OpenAI/openai_api_plugin.py)                                                                         | <img src=Previews/OpenAI.png width=100>                                                                                                                                                     | OpenAI (**API**).</br>Set **api_key:** to your API-Key.</br></br>Provides Speech-to-Text, Text-to-Speech and Text-Translation using OpenAI Cloud Models</br></br>**Not recommended with Realtime mode because that can use up your Credits very quickly.**</br><sub>thx to https://openai.com/ </sub>                                                                                                                                                                                                     | Sharrnah   |
| [**Deepgram API**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/Other-Plugins/Deepgram/deepgram_api_plugin.py)                                                                   | <img src=Previews/Deepgram.png width=100>                                                                                                                                                   | Deepgram (**API**).</br>Set **api_key:** to your API-Key.</br></br>Provides Speech-to-Text and Text-to-Speech using Deepgram Cloud Models</br></br>**Not recommended with Realtime mode because that can use up your Credits very quickly.**</br><sub>thx to https://deepgram.com </sub>                                                                                                                                                                                                                  | Sharrnah   |
| [**Simple Soundboard**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/Other-Plugins/Soundboard/soundboard_plugin.py)                                                              | <img src=Previews/Soundboard.png width=250>                                                                                                                                                 | Provides a simple Soundboard where you can play audio files with a click of a button.</br>Audio-files in sub-folders are grouped together.                                                                                                                                                                                                                                                                                                                                                                | Sharrnah   |
| [**RVC Voice-Conversion Speech 2 Speech**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/Speech-Conversion-Plugins/RVC/rvc_sts_plugin.py)                                         |                                                                                                                                                                                             | Retrieval-based-Voice-Conversion Plugin.</br>Use RVC models to convert:</br>- Your speech (also in Realtime)</br>- Any Text-to-Speech</br>- Speech of audio-files</br>into the models voice.                                                                                                                                                                                                                                                                                                              | Sharrnah   |
| [**Large Language Model Conversation**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/Other-Plugins/LLM-Conversation/llm_plugin.py)                                               | <img src=Previews/LLM-Conversation.png width=250>                                                                                                                                           | Implementation to run Large Language Models together with Whispering Tiger.                                                                                                                                                                                                                                                                                                                                                                                                                               | Sharrnah   |
| [**Show currently playing song over OSC**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/VRChat-Plugins/Currently-Playing-Chatbox/current_playing_plugin.py)                      | <img src=Previews/Currently-Playing-Chatbox.gif width=250>                                                                                                                                  | Displays the Song Title and Author of the Song you are currently listening to in your favourite music player inside VRChat using OSC.                                                                                                                                                                                                                                                                                                                                                                     | Sharrnah   |
| [**Volume and audio direction send over OSC**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/VRChat-Plugins/Audio-Direction-Avatar-Control/audio_direction_plugin.py)             | <img src=Previews/Audio-Direction-Avatar-Control.gif width=250>                                                                                                                             | Add the synced float parameters `audio_volume` and `audio_direction` to your VRChat avatar. <ul><li>`audio_direction` at `/avatar/parameters/audio_direction`: the direction of the sound. Where 0.5 is centered, 0 is left 1 is right.</li> <li>`audio_volume` at `/avatar/parameters/audio_volume`: the volume of the sound. Where 0 is silent, 1 is loud.</li></ul> <sub>_Inspired by https://github.com/Codel1417/VRC-OSC-Audio-Reaction :love_letter:_</sub>                                         | Sharrnah   |
| [**Control VRChat Avatar Parameters by Commands**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/VRChat-Plugins/Command-Avatar-Control/command_control_plugin.py)                 | <img src=Previews/Command-Avatar-Control.gif width=250>                                                                                                                                     | Controls VRChat Avatar Parameters by custom commands.                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Sharrnah   |
| [**Control VRChat Avatar Parameters by Emotion Prediction**](https://github.com/Sharrnah/whispering-plugins/blob/main/Plugins/VRChat-Plugins/Text-Emotion-Avatar-Control/text_emotion_vrc_plugin.py) | <img src=Previews/Text-Emotion-Avatar-Control.gif width=250>                                                                                                                                | Controls VRChat Avatar Parameters by Emotion Prediction.                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Sharrnah   |


## Plugin Documentations
- [RVC Voice Conversion Plugin](Documentation/Speech-Conversion-Plugins/RVC-Plugin.md)