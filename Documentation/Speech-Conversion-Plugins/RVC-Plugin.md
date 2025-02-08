# RVC Voice Conversion Plugin

The RVC Plugin allows to replace one voice with another. This means the input is an existing audio.

(It is not Text-to-Speech which works with Text as Input and generates audio from this.)

You need a RVCv2 Model for this to work.

## Configuration
- Under the __General-Tab__, select the RVCv2 Model (and Index) file you want the voice to be converted into.
    - As `f0method` select `rmvpe` for best quality, or `crepe` (or others) for faster but lower quality.
    - Set `f0up_key` to a value that fits the conversion pitch. This is depending on the Input and Output Voice.
      - lower **(~ -12)** if the voice conversion is female to male
      - higher **(~ +12)** if the voice conversion is male to female
- Under the __Audio conversion-Tab__, select the source from where you want to convert the voice from in `voice_change_source`.
    - `Own Voice` works by using Whispering Tigers audio recording logic. So same VAD, Speech Pause and Timelimit settings apply as when using Speech-to-Text models.
      
      You also need to enable `Speech-to-Text Enabled` and `Automatic Text-to-Speech` Checkboxes in the `Speech-to-Text` Tab of the main application.

      <img src=rvc-main-app-settings.png width=350>

    - `Own Voice (Realtime)` works by starting a seperate thread. All Realtime settings can be found under the Plugin `Realtime-Tab`.

    - Set it to `Text-to-Speech` when using any TTS Model/Plugin to convert the TTS Voice into the RVC Models voice.
- Under the __Model-Tab__,

  *__select the RVCv2 Model (and Index) file you want the voice to be converted into.__*
  - `device` Select your GPU for best performance. If you have one GPU, select `cuda:0` if you have an NVIDIA GPU, or `direct-ml:0` or `direct-ml:1` for AMD/Intel.

    (Try to change this if it is too slow and does not seem to use the correct GPU, or you get the error `'RVC' object has no attribute 'tgt_sr'`)
  - `half_precision` Enable this when using a GPU and to increase speed for slightly less quality and also less memory usage.
- Under the __Realtime-Tab__ _(Only needed in Realtime mode)_
    - `rt_block_time` Is the audio block length. Lower values reduce the delay but increase the GPU usage and possibly lower the quality.
    - `rt_crossfade_time` is the time each block is faded over the next to make it more seamless.
    - `rt_extra_time` is added audio data which can increase quality.
    - `rt_input_device_index` Is the Input Audio device. (should be your microphone. [Using NVIDIA Broadcast can heavily increase quality for higher GPU time])
    - `rt_output_device_index` Is the Output Audio device where the audio is played to. (If you want to use it in voice-chats, you have to select a Virtual Audio Cable here and the same in voice-chat applications. Find a VB-Cable Driver here https://vb-audio.com/Cable/)
    - `rt_threshold` defines the threshold at which volume it records the audio. Too high and it might cut out too much or not record at all. Too low and it might pick up too much noise.

    Press the `start / restart / stop Realtime` Button to manually start the thread or restart when you changed some settings.
    
    (_If you set `voice_change_source` from the _Audio conversion-Tab_ to `Disabled` and press the `start / restart / stop Realtime` Button, it will stop the thread._)


