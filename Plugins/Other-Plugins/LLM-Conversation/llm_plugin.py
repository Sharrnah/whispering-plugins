# ============================================================
# Adds Large Language Model support to Whispering Tiger
# answers to questions using speech to text or if using the TTS send event
# V1.0.0
#
# See https://github.com/Sharrnah/whispering
# ============================================================

import Plugins

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import os
from time import strftime

import VRC_OSCLib
import websocket
import json
import settings
from Models import languageClassification
from Models.TTS import silero
from Models.TextTranslation import texttranslate
import re

import sys
from importlib import util
import zipfile
import downloader

DEFAULT_PROMPT = "This is a discussion between a [human] and a [AI]. \nThe [AI] is very\
  nice and empathetic.\n\n[human]: What color do you like?\n[AI]: I like pink.\n\
  \n[human]: Hello nice to meet you.\n[AI]: Nice to meet you too.\n\n[human]: What\
  are you? \n[AI]: I am an AI.\n\n[human]: Who created you?\n[AI]: Sharrnah created\
  me. Check https://github.com/Sharrnah/whispering.\n\n[human]: How is it going today?\n[AI]: Not so bad, thank\
  you! How about you?\n\n[human]: I am okay too. \n[AI]: Oh that's good.\n\n[human]:\
  ??\n[AI]: "

PROMPT_FORMATTING = {
    "question": ["about ", "across ", "after ", "against ", "along ", "am ", "amn't ", "among ", "are ", "aren't ", "around ", "at ", "before ", "behind ", "between ",
                 "beyond ", "but ", "by ", "can ", "can't ", "concerning ", "could ", "couldn't ", "despite ", "did ", "didn't ", "do ", "does ", "doesn't ", "don't ",
                 "down ", "during ", "except ", "following ", "for ", "from ", "had ", "hadn't ", "has ", "hasn't ", "have ", "haven't ", "how ", "how's ", "in ",
                 "including ", "into ", "is ", "isn't ", "like ", "may ", "mayn't ", "might ", "mightn't ", "must ", "mustn't ", "near ", "of ", "off ", "on ", "out ",
                 "over ", "plus ", "shall ", "shan't ", "should ", "shouldn't ", "since ", "through ", "throughout ", "to ", "towards ", "under ", "until ", "up ", "upon ",
                 "was ", "wasn't ", "were ", "weren't ", "what ", "what's ", "when ", "when's ", "where ", "where's ", "which ", "which's ", "who ", "who's ", "why ",
                 "why's ", "will ", "with ", "within ", "without ", "won't ", "would ", "wouldn't "],
    "command": ["ai? ", "ai. ", "ai ", "a.i. ", "ai, ", "ai! ", "artificial intelligence"],
}

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


def extract_zip(file_path, output_dir):
    with zipfile.ZipFile(file_path, "r") as zip_file:
        zip_file.extractall(path=output_dir)
    # remove the zip file after extraction
    os.remove(file_path)


bitsandbytes_dependency_module = {
    "url": "https://files.pythonhosted.org/packages/0d/95/04de4035b1874026cadb9219aafb890e14fc6f3702a67618a144c66ec710/bitsandbytes-0.38.1-py3-none-any.whl",
    "sha256": "5f532e7b1353eb7049ae831da2eb62ed8a1e0444116bd51b9e088a6e0bc7a34a",
    "path": "bitsandbytes"
}

llm_plugin_dir = Path(Path.cwd() / "Plugins" / "llm_plugin")
os.makedirs(llm_plugin_dir, exist_ok=True)
llm_cache_dir = Path(Path.cwd() / ".cache" / "llm_plugin")
os.makedirs(llm_cache_dir, exist_ok=True)

def sanitize_folder_name(folder_name):
    """
    Replaces characters that are not supported in Windows folder names with an underscore.
    """
    # Define a regular expression to match characters that are not allowed in Windows folder names
    illegal_char_pattern = re.compile(r'[<>:"/\\|?*]')

    # Replace any illegal characters with an underscore
    sanitized_name = illegal_char_pattern.sub('_', folder_name)

    return sanitized_name


class LlmPlugin(Plugins.Base):
    tokenizer = None
    model = None
    model_name = "EleutherAI/gpt-j-6B"
    bit_length = 16  # can be 32 = 32 float, 16 = 16 float or 8 = 8 int
    device = "cpu"  # can be "auto" or None
    low_cpu_mem_usage = True
    load_in_8bit_mode = False
    max_new_tokens = 2048

    conditioning_lines = []

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                # General
                "model_name": "EleutherAI/gpt-j-6B", # the huggingface model name. Good alternatives are "bigscience/bloomz-7b1", "bigscience/bloom-7b1", "PygmalionAI/pygmalion-6b", "google/flan-t5-large", "mosaicml/mpt-7b-chat", mistralai/Mistral-7B-Instruct-v0.1 ...
                "device": {"type": "select", "value": "auto", "values": ["auto", "cpu", "mps", "cuda"]},
                "bit_length": {"type": "select", "value": "16", "values": ["32", "16", "8"]},
                "load_in_8bit_mode": False,
                "max_new_tokens": 2048,
                "prompt_prefix": DEFAULT_PROMPT, # replaces ?? in prompt with input text or adds it to the end if no ?? is found
                "conditioning_history": 0, # number of lines to add from previous conversation
                "memory": "", # long term memory to add to the conversation
                "osc_prefix": "AI: ",
                "translate_to_speaker_language": False,
                "only_respond_question_commands": False,
                "tts_enabled": False,
                "osc_enabled": True,
                "is_instruct": False,
            },
            settings_groups={
                "General": ["max_new_tokens", "prompt_prefix", "conditioning_history", "memory", "osc_prefix", "translate_to_speaker_language", "only_respond_question_commands", "tts_enabled", "osc_enabled"],
                "Model": ["model_name", "device", "bit_length", "load_in_8bit_mode", "is_instruct"],
            }
        )

        self.model_name = self.get_plugin_setting("model_name", "EleutherAI/gpt-j-6B")  # the huggingface model name. Good alternatives are "bigscience/bloomz-7b1", "bigscience/bloom-7b1", "PygmalionAI/pygmalion-6b" ...
        self.bit_length = int(self.get_plugin_setting("bit_length", 16))  # can be 32 = 32 float, 16 = 16 float or 8 = 8 int
        self.load_in_8bit_mode = self.get_plugin_setting("load_in_8bit_mode", False)
        self.max_new_tokens = self.get_plugin_setting("max_new_tokens", 2048)

        if self.is_enabled(False):
            # load the bitsandbytes module
            if self.load_in_8bit_mode:
                if not Path(llm_plugin_dir / bitsandbytes_dependency_module["path"] / "__init__.py").is_file():
                    downloader.download_thread(bitsandbytes_dependency_module["url"], str(llm_plugin_dir.resolve()),
                                               bitsandbytes_dependency_module["sha256"])
                    extract_zip(str(llm_plugin_dir / os.path.basename(bitsandbytes_dependency_module["url"])),
                                str(llm_plugin_dir.resolve()))

                # add cuda dlls to path
                if not Path(llm_plugin_dir / bitsandbytes_dependency_module["path"] / "cuda_setup" / "libbitsandbytes_cuda116.dll").is_file():
                    downloader.download_thread("https://github.com/Keith-Hon/bitsandbytes-windows/raw/main/bitsandbytes/cuda_setup/libbitsandbytes_cuda116.dll", Path(llm_plugin_dir / bitsandbytes_dependency_module["path"] / "cuda_setup").resolve(), None)

                bitsandbytes = load_module(
                    str(Path(llm_plugin_dir / bitsandbytes_dependency_module["path"]).resolve()))

            cache_path = Path(llm_cache_dir / sanitize_folder_name(self.model_name))
            os.makedirs(cache_path, exist_ok=True)
            print("llm cache folder: " + str(cache_path.resolve()))

            if self.model is None:
                print(f"{self.model_name} is Loading to {self.device} using {self.bit_length} bit {('INT' if self.bit_length == 8 else 'float')} precision...")
                websocket.set_loading_state("llm_loading", True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                if self.device == "auto" or self.device == "cuda":
                    device_map = "auto"
                else:
                    device_map = {"": self.device}

                match self.bit_length:
                    case 16:  # 16 bit float
                        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                                          cache_dir=str(cache_path.resolve()),
                                                                          trust_remote_code=True,
                                                                          revision="float16",
                                                                          device_map=device_map, load_in_8bit=self.load_in_8bit_mode,
                                                                          torch_dtype=torch.float16,
                                                                          low_cpu_mem_usage=self.low_cpu_mem_usage)
                    case 8:  # 8 bit int
                        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                                          cache_dir=str(cache_path.resolve()),
                                                                          trust_remote_code=True,
                                                                          device_map=device_map, load_in_8bit=self.load_in_8bit_mode,
                                                                          low_cpu_mem_usage=self.low_cpu_mem_usage)
                    case _:  # 32 bit float
                        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                                          cache_dir=str(cache_path.resolve()),
                                                                          trust_remote_code=True,
                                                                          device_map=device_map, load_in_8bit=self.load_in_8bit_mode,
                                                                          low_cpu_mem_usage=self.low_cpu_mem_usage)

                if not self.load_in_8bit_mode:
                    self.model.half()

                self.model.eval()
                if torch.__version__ >= "2" and sys.platform != "win32":
                    self.model = torch.compile(self.model)

                websocket.set_loading_state("llm_loading", False)
                # load text translator
                #texttranslate.InstallLanguages()

            # disable OSC processing so the LLM can take it over:
            settings.SetOption("osc_auto_processing_enabled", False)
            # disable TTS so the LLM can take it over:
            settings.SetOption("tts_answer", False)
            # disable websocket final messages processing so the LLM can take it over:
            settings.SetOption("websocket_final_messages", False)
        pass

    def encode(self, input_text, retry=0):
        original_input_text = input_text

        # show typing indicator when processing
        osc_ip = settings.GetOption("osc_ip")
        osc_port = settings.GetOption("osc_port")
        if self.get_plugin_setting("osc_enabled", True) and osc_ip != "0":
            VRC_OSCLib.Bool(True, "/chatbox/typing", IP=osc_ip, PORT=osc_port)

        # make sure input has an end token
        if not input_text.endswith(".") and not input_text.endswith("!") and not input_text.endswith(
                "?") and not input_text.endswith(",") and not input_text.endswith(";") and not input_text.endswith(":"):
            input_text += "."

        # Add llm prompt prefix
        if self.get_plugin_setting("prompt_prefix", "") != "":
            llm_prompt_prefix = self.get_plugin_setting("prompt_prefix", "")
            if llm_prompt_prefix.count("??") > 0:
                input_text = llm_prompt_prefix.replace("??", input_text)
            else:
                input_text = llm_prompt_prefix + input_text
        conditioning_input_text = input_text

        # add current time infos
        input_text = strftime("It is %A the %d %B %Y and the time is %H:%M.") + "\n" + input_text

        # Add conditioning lines
        if self.get_plugin_setting("conditioning_history", 0) > 0 and len(self.conditioning_lines) > 0:
            input_text = "\n".join(self.conditioning_lines) + "\n" + input_text

        # Add llm long-term memory
        if self.get_plugin_setting("memory", "") != "":
            input_text = self.get_plugin_setting("memory") + "\n" + input_text

        if self.get_plugin_setting("is_instruct"):
            input_ids = self.tokenizer.apply_chat_template(input_text, return_tensors="pt")
        else:
            input_ids = self.tokenizer(input_text, return_tensors="pt")
        if 'input_ids' in input_ids:
            input_ids = input_ids['input_ids']

        input_ids.to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                input_ids,
                do_sample=True,
                use_cache=True,
                temperature=0.8,
                min_length=len(input_ids[0]) + 10,
                #max_length=len(input_ids[0]) + 40,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                max_new_tokens=self.max_new_tokens
            )

        result = self.tokenizer.batch_decode(output_tokens)[0]

        result = result.replace("<pad>", "").replace("</s>", "").replace("<unk>", "").strip()

        # remove the input text and human hallucination from the result
        ol = len(input_text)
        n = 0
        for i in range(ol, len(result)):
            if result[i] == '\n\n' or result[i] == '[':
                n = i
                break
        result = result[ol:n]

        # remove some common prefixes from the start of the result (@todo: make this configurable)
        result = result.strip().removeprefix(self.get_plugin_setting("memory"))
        result = result.strip().removeprefix("\n".join(self.conditioning_lines) + "\n")
        result = result.strip().removeprefix(conditioning_input_text)

        result = result.removeprefix("A: ")
        result = result.removeprefix("AI: ")
        result = result.removeprefix("Human: ")
        result = result.removeprefix("[human]")
        result = result.removeprefix(":")

        if result.strip() == "":
            if retry < 3:
                return self.encode(original_input_text, retry + 1)
            else:
                result = "hmm..."

        # Add the result to the conditioning history and remove the oldest lines if needed
        if self.get_plugin_setting("conditioning_history", 0) > 0:
            if len(self.conditioning_lines) >= self.get_plugin_setting("conditioning_history"):
                difference = len(self.conditioning_lines) - self.get_plugin_setting("conditioning_history")
                del self.conditioning_lines[0:difference - 1]

            self.conditioning_lines.append(conditioning_input_text + result)
        else:
            self.conditioning_lines.clear()

        return result.strip()

    def send_message(self, text, answer, result_obj):
        osc_ip = settings.GetOption("osc_ip")
        osc_address = settings.GetOption("osc_address")
        osc_port = settings.GetOption("osc_port")
        llm_osc_prefix = self.get_plugin_setting("osc_prefix", "AI: ")

        result_obj["type"] = "llm_answer"
        try:
            print("LLM Answer: " + answer)
        except:
            print("LLM Answer: ???")

        if self.get_plugin_setting("osc_enabled", True) and answer != text and osc_ip != "0":
            VRC_OSCLib.Chat(llm_osc_prefix + answer, True, True, osc_address, IP=osc_ip, PORT=osc_port,
                            convert_ascii=settings.GetOption("osc_convert_ascii"))

        websocket.BroadcastMessage(json.dumps(result_obj))

        if self.get_plugin_setting("tts_enabled", False) and answer != "" and silero.init():
            # remove osc prefix from message
            predicted_text = answer.removeprefix(llm_osc_prefix).strip()
            try:
                silero_wav, sample_rate = silero.tts.tts(predicted_text)
                silero.tts.play_audio(silero_wav, settings.GetOption("device_out_index"))
            except Exception as e:
                print("Error while playing TTS audio: " + str(e))

    def timer(self):
        pass

    def stt(self, text, result_obj):
        if self.model is not None and self.is_enabled(False):
            # only respond to questions or commands if the setting is enabled
            if (self.get_plugin_setting("only_respond_question_commands") and (("?" in text.strip().lower() and any(ele in text.strip().lower() for ele in PROMPT_FORMATTING['question'])) or
                                                                                      any(ele in text.strip().lower() for ele in PROMPT_FORMATTING['command']))) or \
                    not self.get_plugin_setting("only_respond_question_commands"):
                predicted_text = self.encode(text)

                if self.get_plugin_setting("translate_to_speaker_language", False):
                    target_lang = result_obj['language']
                    print("Translating to " + target_lang)
                    predicted_text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(predicted_text, "auto",
                                                                                                 target_lang,
                                                                                                 False, True)
                result_obj['llm_answer'] = predicted_text

                print("llm_answer: ", predicted_text)

                self.send_message(text, predicted_text, result_obj)
        return

    def tts(self, text, device_index, websocket_connection=None, download=False):
        if self.model is not None and self.is_enabled(False):
            predicted_text = self.encode(text)

            # detect written text language
            language = languageClassification.classify(text)

            result_obj = {'text': text, 'type': "transcribe", 'language': language, 'llm_answer': predicted_text}

            self.send_message(text, predicted_text, result_obj)

        return

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        settings.SetOption("websocket_final_messages", True)

        #self.model = None
        #self.tokenizer = None

        #if torch.cuda.is_available():
        #    # Reset the maximum memory allocated by PyTorch
        #    torch.cuda.reset_max_memory_allocated()

        #    # Empty the GPU memory cache
        #    torch.cuda.empty_cache()
