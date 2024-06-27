# Plugins

Current list of plugins that are available for [Whispering Tiger](https://github.com/Sharrnah/whispering).

_If you have created a plugin, please add it to this list using a pull request or let me know and I will add it._

_See [plugin-creation.md](Documentation/plugin-creation.md) for more information on how to create plugins yourself._

### Install
Copy the plugin file to the `Plugins` directory in the root of the project.

Consider reading the plugin file to see if there are any dependencies that need to be installed.

## List of Plugins

| Title                                                                                                                           | Preview                                                                                                                                                                               | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Author     |
|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| [**Keyboard Typing**](https://gist.github.com/Sharrnah/2da117d967eef6bc49689d15eb622f80)                                        | <img src=Previews/Keyboard-Typing.gif width=250>                                                             | Type Text without your keyboard but instead with your Voice. <br> supports additional _customizable_ commands like: <br> - start typing<br>- stop typing<br>- new line                                                                                                                                                                                                                                                                                                                                    | Sharrnah   |
| [**Subtitle Display**](https://gist.github.com/Sharrnah/6ac98143d4fa7bfd3867e57be6a0572a)                                       | <img src=https://user-images.githubusercontent.com/55756126/236357319-8769c88d-f9bb-492c-8be8-89a20e521792.gif width=250>                                                             | Display Subtitles everywhere on your Desktop                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Sharrnah   |
| [**Subtitle Export**](https://gist.github.com/Sharrnah/f3e35fdfed1779e80ca5b1706e03ecf6)                                        | <img src=https://user-images.githubusercontent.com/55756126/270682617-4afe408b-923d-44a1-8a76-11d3a87c9270.png width=250>                                                             | Generate Subtitle files for Audio or Video files.<br>_Can export as .VTT, .SRT or .SBV_                                                                                                                                                                                                                                                                                                                                                                                                                   | Sharrnah   |
| [**Voicevox Text 2 Speech**](https://gist.github.com/Sharrnah/7071f08d539bba6bd18e15ca40fc7c47)                                 | <video src='https://user-images.githubusercontent.com/55756126/232867089-5154c472-1c0b-4f20-acba-5a5d869b775e.mp4' width=100>                                                         | Japanese Text 2 Speech.</br>change **speaker** of selected model in Text-to-Speech tab.</br>**acceleration_mode:** can be "CPU" or "CUDA" </br><sub>thx to https://voicevox.hiroshiba.jp/ </sub>                                                                                                                                                                                                                                                                                                          | Sharrnah   |
| [**TALQu3PRO Text 2 Speech**](https://github.com/rokujyushi/WT2TALQu/blob/main/TALQu3PRO_TTS.py)                                |                                                                                                                                                                                       | Japanese Text 2 Speech.</br>Set the path to TALQuClient in General.</br><sub>thx to https://haruqa.github.io/TALQu/ </sub>                                                                                                                                                                                                                                                                                                                                                                                | Rokujyushi |
| [**Bark Text 2 Speech**](https://gist.github.com/Sharrnah/5b19b4a7fa22d43c503c33d24c85e778)                                     | <video src='https://user-images.githubusercontent.com/55756126/253620129-f7859d83-9221-4f03-9eeb-d8ec3afde702.mp4' width=100>                                                         | Multilingual Text 2 Speech</br>change **history_prompt:** to one of the voices you can find here: [Bark Speaker Library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683) </br>**prompt_wrap** Setting can be used for more prompt features like:</br>- singing ("`♪ ## ♪`")</br>- bias towards male or female ("`[MAN] ##`" or "`[WOMAN] ##`")</br>- more infos in their [Readme](https://github.com/suno-ai/bark#-usage-in-python).</br><sub>thx to https://github.com/suno-ai/bark </sub> | Sharrnah   |
| [**Coqui Text 2 Speech**](https://gist.github.com/Sharrnah/7c78f4469a01da0c706f6d4adb6a88fc)                                    | _example generated with `tts_models/en/vctk/vits`_</br> <video src='https://user-images.githubusercontent.com/55756126/256297547-a1c27c22-2f69-45ff-b61d-66f27f7c9cc9.mp4' width=100> | Multilingual Text 2 Speech</br> Supports many different TTS Models, including:</br>- Bark</br>- Tortoise v2</br>- VITS</br>- fairseq VITS (with ~1100 languages)</br>- and many more.</br>In addition it features a one-shot Voice Conversion model **FreeVC** which can be used with Text 2 Speech or Speech 2 Speech.</br><sup>_(Plugin uses a locally running Coqui Server)_</sup></br><sub>thx to https://github.com/coqui-ai/TTS/ </sub>                                                             | Sharrnah   |
| [**ElevenLabs Text 2 Speech**](https://gist.github.com/Sharrnah/b036126ac1013af1fc625091cf02eac8)                               | <video src='https://user-images.githubusercontent.com/55756126/236304921-a64f0443-ac45-4181-bc53-090696a58f0b.mp4' width=100>                                                         | Multilingual Text 2 Speech (**API**).</br>Set **api_key:** to your API key.</br>change **voice:** to one of voices and **voice_index:** to the index of the voice. _(other than 0 if more voices with same name exist)_</br>**stt_*:** Settings can limit the generation to prevent accidental use up of available chars on your account.</br><sub>thx to https://elevenlabs.io/ </sub>                                                                                                                   | Sharrnah   |
| [**DeepL Text Translation**](https://gist.github.com/Sharrnah/39e7f924d0af5b4b2bbbfb7fe21d3e50)                                 | <img src=https://raw.githubusercontent.com/Sharrnah/whispering/main/images/docs/DeepL.png width=100>                                                                                  | DeepL Text Translation (**API**).</br>Set **auth_key:** to your Authentication key.</br></br>Be careful using it with realtime mode, as it might use up your characters fast.</br><sub>thx to https://www.deepl.com/ </sub>                                                                                                                                                                                                                                                                               | Sharrnah   |
| [**Simple Soundboard**](https://gist.github.com/Sharrnah/a1a53c2e2de3148d8b24fa6749691e98)                                      | <img src=https://user-images.githubusercontent.com/55756126/260482610-3aaddc7d-0f75-4cf5-b14e-b6e805ff1fa5.png width=250>                                                             | Provides a simple Soundboard where you can play audio files with a click of a button.</br>Audio-files in sub-folders are grouped together.                                                                                                                                                                                                                                                                                                                                                                | Sharrnah   |
| [**RVC Voice-Conversion Speech 2 Speech**](https://gist.github.com/Sharrnah/8d906a3657f097702079451ff762ed95)                   |                                                                                                                                                                                       | Retrieval-based-Voice-Conversion Plugin.</br>Use RVC models to convert your speech or speech of audio files into the models voice.                                                                                                                                                                                                                                                                                                                                                                        | Sharrnah   |
| [**Large Language Model Answering**](https://gist.github.com/Sharrnah/eeaf2acda3e92d8eed1747f05a3f4102)                         | <img src=https://user-images.githubusercontent.com/55756126/225940740-f5e44911-9836-4b26-ab6e-a32676ddd27e.png width=250>                                                             | Implementation to run Large Language Models together with Whispering Tiger.                                                                                                                                                                                                                                                                                                                                                                                                                               | Sharrnah   |
| [**Show currently playing song over OSC**](https://gist.github.com/Sharrnah/802ab486374c69a183c85d5846100232)                   | <img src=https://user-images.githubusercontent.com/55756126/223178202-ef31fb96-6fa8-4427-9f5e-b4dd587f07ab.png width=250>                                                             | Displays the Song Title and Author of the Song you are currently listening to in your favourite music player inside VRChat using OSC.                                                                                                                                                                                                                                                                                                                                                                     | Sharrnah   |
| [**Volume and audio direction send over OSC**](https://gist.github.com/Sharrnah/582b8a390e2462bcec77332cac2eb570)               | <img src=https://user-images.githubusercontent.com/55756126/228648156-56de7f87-476a-4569-866a-8b8591b2549e.gif width=250>                                                             | Add the synced float parameters `audio_volume` and `audio_direction` to your VRChat avatar. <ul><li>`audio_direction` at `/avatar/parameters/audio_direction`: the direction of the sound. Where 0.5 is centered, 0 is left 1 is right.</li> <li>`audio_volume` at `/avatar/parameters/audio_volume`: the volume of the sound. Where 0 is silent, 1 is loud.</li></ul> <sub>_Inspired by https://github.com/Codel1417/VRC-OSC-Audio-Reaction :love_letter:_</sub>                                         | Sharrnah   |
| [**Control VRChat Avatar Parameters by Commands**](https://gist.github.com/Sharrnah/64ea762819b39c5bddbac2730ae43dcc)           | <img src=https://user-images.githubusercontent.com/55756126/228892285-a2148a33-94b2-460c-9632-423f77235c03.gif width=250>                                                             | Controls VRChat Avatar Parameters by custom commands.                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Sharrnah   |
| [**Control VRChat Avatar Parameters by Emotion Prediction**](https://gist.github.com/Sharrnah/28564fd26cef6f1689ea5fc3053b7ee2) | <img src=https://user-images.githubusercontent.com/55756126/229387209-c8943a7a-9f51-4206-babb-239925d0ace7.gif width=250>                                                             | Controls VRChat Avatar Parameters by Emotion Prediction.                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Sharrnah   |
