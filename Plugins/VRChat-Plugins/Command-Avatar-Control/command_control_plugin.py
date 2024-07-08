# ============================================================
# Plugin to listen for commands to send OSC messages using Whispering Tiger
# Used to Control Avatar Parameters in VRChat
# Version: 1.1.3
# See https://github.com/Sharrnah/whispering
# ============================================================

import math
import pickle
import socket
import time

import Plugins

import os
import yaml
import VRC_OSCLib
import settings

import threading
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

from typing import List, Any

# Commands are defined as a tuple of words, and a dictionary of OSC parameters and values.
# special values are "toggle" and "toggle_int=1"
# "toggle" toggles between bool True and False.
# "toggle_int=1" toggles an integer to the value after the '='. if it is toggled off, it will be set to 0.
#
# The yaml file should contain a list of commands, each with a list of words like this:
#
# - command: ["enable", "white"]
#   param: "WhiteTiger"
#   value: 1
#
# - command: ["disable", "white"]
#   param: "WhiteTiger"
#   value: 0
#
# - command: ["toggle", "white"]
#   param: "WhiteTiger"
#   value: "toggle_int=1"
#
# - command: ["toggle", "shirt"]
#   param: "shirt"
#   value: "toggle"
#

COMMANDS = {
    ("toggle", "shirt"): {"param": "shirt", "value": "toggle"},
}

MEMORY = {}

PUBLISHER_PORT = 5553


class CommandControlPlugin(Plugins.Base):
    commands = None
    server_thread = None

    def publisher_thread(self, server_socket):
        global MEMORY
        while True:
            client_socket, _ = server_socket.accept()
            while True:
                try:
                    serialized_memory = pickle.dumps(MEMORY)
                    client_socket.sendall(serialized_memory)
                    client_socket.sendall(b'\n')  # Add a newline character as a delimiter
                    time.sleep(1)  # Adjust the sleep time as needed
                except (BrokenPipeError, ConnectionResetError):
                    print("Client disconnected")
                    break
            client_socket.close()

    def request_memory_thread(self):
        global MEMORY
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(("127.0.0.1", PUBLISHER_PORT))
        while True:
            received_data = bytearray()
            while True:
                try:
                    chunk = client_socket.recv(4096)
                except (BrokenPipeError, ConnectionResetError):
                    if self.get_plugin_setting("debug", False):
                        print("Server disconnected")
                    break
                if not chunk:
                    break
                received_data.extend(chunk)
                if b'\n' in received_data:
                    break

            MEMORY = pickle.loads(received_data.rstrip(b'\n'))
            if self.get_plugin_setting("debug", False):
                print("Received memory:", MEMORY)
            time.sleep(1)  # Adjust the sleep time according to your needs
        #client_socket.close()

    def _receive_osc_parameters(self, address: str, *args: List[Any]) -> None:
        # Check that address starts with filter
        if not address.startswith("/avatar/parameters/"):
            return

        param_name = address[len("/avatar/parameters/"):]
        value = args[0]

        # Check if the parameter is in the COMMANDS variable
        param_in_commands = False
        for command in self.commands.values():
            if command["param"] == param_name:
                param_in_commands = True
                break

        # Update the MEMORY variable with the new value
        if param_in_commands:
            MEMORY[param_name] = value

            if self.get_plugin_setting("debug", False):
                print(f"Received OSC parameter {param_name} with value {value}")

    def osc_server(self):
        osc_ip = settings.GetOption("osc_ip")
        osc_server_port = self.get_plugin_setting("osc_server_port", 9001)

        try:
            # start MEMORY publisher
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind(("127.0.0.1", PUBLISHER_PORT))
            server_socket.listen(1)
            server_thread = threading.Thread(target=self.publisher_thread, args=(server_socket,))
            server_thread.start()

            # start OSC server
            dispatcher = Dispatcher()
            dispatcher.map("/avatar/parameters/*", self._receive_osc_parameters)
            server = osc_server.ThreadingOSCUDPServer((osc_ip, osc_server_port), dispatcher)
            print(f"OSC Server started on {server.server_address}")
            server.serve_forever()
        except OSError:
            # start MEMORY subscriber if OSC server could not be started
            threading.Thread(target=self.request_memory_thread).start()
            print(f"Could not start OSC Server on {osc_ip}:{osc_server_port}, subscribing to MEMORY instead")

    def jaccard_similarity(self, a, b):
        a, b = set(a), set(b)
        intersection = a.intersection(b)
        union = a.union(b)
        return len(intersection) / len(union) if len(union) > 0 else 0

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

    def cosine_similarity(self, a, b):
        dot_product = sum(p * q for p, q in zip(a, b))
        magnitude_a = math.sqrt(sum(p * p for p in a))
        magnitude_b = math.sqrt(sum(q * q for q in b))
        return dot_product / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0

    def word_to_vector(self, word):
        return [ord(c) for c in word]

    def send_osc_command(self, command_parameter, command_value):
        osc_ip = settings.GetOption("osc_ip")
        osc_port = settings.GetOption("osc_port")

        if isinstance(command_value, float):
            MEMORY[command_parameter] = command_value
            VRC_OSCLib.AV3_SetFloat(command_value, command_parameter, osc_ip, osc_port)
        elif isinstance(command_value, int):
            MEMORY[command_parameter] = command_value
            VRC_OSCLib.AV3_SetInt(command_value, command_parameter, osc_ip, osc_port)
        elif isinstance(command_value, bool):
            MEMORY[command_parameter] = command_value
            VRC_OSCLib.AV3_SetBool(command_value, command_parameter, osc_ip, osc_port)
        elif isinstance(command_value, str) and command_value == "toggle":
            if command_parameter in MEMORY:
                command_value = not MEMORY[command_parameter]
            else:
                command_value = True
            MEMORY[command_parameter] = command_value
            VRC_OSCLib.AV3_SetBool(command_value, command_parameter, osc_ip, osc_port)
        elif isinstance(command_value, str) and command_value.startswith("toggle_int"):
            command_value = int(command_value.split("=")[1])
            if command_parameter in MEMORY:
                if command_value == MEMORY[command_parameter]:
                    command_value = 0
                else:
                    command_value = command_value
            MEMORY[command_parameter] = command_value
            VRC_OSCLib.AV3_SetInt(command_value, command_parameter, osc_ip, osc_port)

        print(f"Command parameter {command_parameter} value {command_value} sent.")

    def _search_word(self, command_text):
        # search for command words in command list
        for command in self.commands:
            if all(word in command_text for word in command):
                # send OSC command
                return True, self.commands[command]["param"], self.commands[command]["value"]

        return False, None, None

    def _search_word_levenshtein(self, command_text, threshold=2, word_threshold=1):
        best_match = None
        best_match_score = float('inf')

        # search for command words in command list
        for command in self.commands:
            command_score = 0
            all_words_found = True
            for word in command:
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
            return True, self.commands[best_match]["param"], self.commands[best_match]["value"]

        return False, None, None

    def _search_word_jaccard(self, command_text, threshold=0.5, word_threshold=0.8):
        best_match = None
        best_match_score = 0

        # search for command words in command list
        for command in self.commands:
            command_score = 0
            all_words_found = True
            for word in command:
                max_word_score = max([self.jaccard_similarity(word, txt_word) for txt_word in command_text])
                if max_word_score < word_threshold:  # You can adjust the threshold for individual words.
                    all_words_found = False
                    break
                command_score += max_word_score
            if not all_words_found:
                continue

            command_score /= len(command)

            if command_score > best_match_score:
                best_match = command
                best_match_score = command_score

        similarity_threshold = threshold  # You can adjust the threshold to your liking.

        if best_match_score >= similarity_threshold:
            # send OSC command
            self.send_osc_command(self.commands[best_match]["param"], self.commands[best_match]["value"])
            return True

        if self.get_plugin_setting("debug", False):
            print(f"Command for {command_text} not found.")

    def _search_word_cosine_vector(self, command_text, threshold=0.8, word_threshold=0.8):
        best_match = None
        best_match_score = 0

        # search for command words in command list
        for command in self.commands:
            command_score = 0
            all_words_found = True
            for word in command:
                max_word_score = max(
                    [self.cosine_similarity(self.word_to_vector(word), self.word_to_vector(txt_word)) for txt_word in
                     command_text])
                if max_word_score < word_threshold:  # You can adjust the threshold for individual words.
                    all_words_found = False
                    break
                command_score += max_word_score
            if not all_words_found:
                continue

            command_score /= len(command)

            if command_score > best_match_score:
                best_match = command
                best_match_score = command_score

        similarity_threshold = threshold  # You can adjust the threshold to your liking.

        if best_match_score >= similarity_threshold:
            return True, self.commands[best_match]["param"], self.commands[best_match]["value"]

        return False, None, None

    def command_handler(self, command_text):
        similarity_algo = self.get_plugin_setting("word_similarity_algo", "levenshtein")
        levenshtein_threshold = self.get_plugin_setting("levenshtein_max_distance_threshold", 1)
        levenshtein_word_threshold = self.get_plugin_setting("levenshtein_max_distance_word_threshold", 1)
        jaccard_threshold = self.get_plugin_setting("jaccard_similarity_threshold", 0.5)
        jaccard_word_threshold = self.get_plugin_setting("jaccard_similarity_word_threshold", 0.5)
        cosine_vector_threshold = self.get_plugin_setting("cosine_vector_similarity_threshold", 0.8)
        cosine_vector_word_threshold = self.get_plugin_setting("cosine_vector_similarity_word_threshold", 0.8)

        # remove punctuations
        command_text = command_text.replace(".", "").replace(",", "").replace("!", "").replace("?", "")
        # split command into words
        command_text = command_text.strip().lower().split(" ")
        # remove empty strings
        command_text = list(filter(None, command_text))

        # search for words in command list
        if similarity_algo == "levenshtein":
            found, command_parameter, command_value = self._search_word_levenshtein(command_text, levenshtein_threshold,
                                                                                    levenshtein_word_threshold)
        elif similarity_algo == "jaccard":
            found, command_parameter, command_value = self._search_word_jaccard(command_text, jaccard_threshold,
                                                                                jaccard_word_threshold)
        elif similarity_algo == "cosine_vector":
            found, command_parameter, command_value = self._search_word_cosine_vector(command_text,
                                                                                      cosine_vector_threshold,
                                                                                      cosine_vector_word_threshold)
        else:
            found, command_parameter, command_value = self._search_word(command_text)

        # send OSC command
        if found:
            self.send_osc_command(command_parameter, command_value)

        if self.get_plugin_setting("debug", False) and not found:
            print(f"Command for {command_text} not found.")

        return found

    def save_commands(self, commands, file_name):
        # define a custom YAML representer function for the "command" key
        def command_representer(dumper, data):
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

        # add the custom representer to the YAML dumper
        yaml.add_representer(tuple, command_representer)

        # convert the keys to lists
        commands_list = [dict(command=tuple(key), param=value["param"], value=value["value"]) for key, value in
                         commands.items()]

        # write the commands to a YAML file
        with open(file_name, "w") as f:
            yaml.dump(commands_list, f, default_flow_style=False)

    def load_commands(self):
        commands_file = self.get_plugin_setting("commands_file", "plugin_commands.conf")

        # check if the YAML file exists
        if not os.path.exists(commands_file):
            self.save_commands(COMMANDS, commands_file)

        # load the YAML file
        with open(commands_file, "r") as f:
            commands_yaml = yaml.safe_load(f)

        # convert the YAML data into the desired format
        self.commands = {}
        for command_yaml in commands_yaml:
            command = tuple(command_yaml["command"])
            self.commands[command] = {
                "param": command_yaml["param"],
                "value": command_yaml["value"]
            }

    def init(self):
        self.init_plugin_settings(
            {
                "debug": False,
                "osc_server_port": 9001,
                "levenshtein_max_distance_threshold": 1,
                "levenshtein_max_distance_word_threshold": 1,
                "jaccard_similarity_threshold": 0.5,
                "jaccard_similarity_word_threshold": 0.5,
                "cosine_vector_similarity_threshold": 0.8,
                "cosine_vector_similarity_word_threshold": 0.8,

                "word_similarity_algo": {"type": "select", "value": "levenshtein", "values": ["NONE", "levenshtein", "jaccard", "cosine_vector"]},
            }
        )


        self.load_commands()
        if self.is_enabled(False):
            # start OSC server
            if self.server_thread is None:
                self.server_thread = threading.Thread(target=self.osc_server)
                self.server_thread.start()
        else:
            # stop OSC server
            if self.server_thread is not None:
                self.server_thread.join()
                self.server_thread = None

    def stt(self, text, result_obj):
        if self.is_enabled(False):
            self.command_handler(text)
        return

    def on_enable(self):
        self.init()
        pass
