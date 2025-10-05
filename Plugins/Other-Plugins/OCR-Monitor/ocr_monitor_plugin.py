# ============================================================
# OCR Monitor plugin for Whispering Tiger
# Version: 0.0.3
# This will monitor a region of the screen for text and send it to Whispering Tiger for processing.
# ============================================================
import json
import os
import platform
import shutil
import traceback
from pathlib import Path

import downloader
import settings
import websocket
from Models import OCR
from Models.TTS import tts
from Models.TextTranslation import texttranslate

if platform.system() == 'Windows':
    import mss

import numpy as np
from PIL import Image

import Plugins
import tkinter as tk
import threading
import queue
import time


from importlib import util
import importlib
import pkgutil
import sys

def load_module(package_dir, recursive=False):
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

    if recursive:
        # Recursively load all submodules
        for _, name, _ in pkgutil.walk_packages(module.__path__, module.__name__ + '.'):
            importlib.import_module(name)

    # Remove the parent directory from sys.path
    sys.path.pop(0)

    return module


DEPENDENCY_LINKS = {
    "tesseract": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:projects/tesseract/tesseract-5.4.0.20240606.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:projects/tesseract/tesseract-5.4.0.20240606.zip",
            "https://s3.libs.space:9000/projects/tesseract/tesseract-5.4.0.20240606.zip",
        ],
        "checksum": "0cb8f4e6abe44097e27e89409290babc636b347421c139a5b266aef0e7791198",
        "file_checksums": {},
        "path": "tesseract",
    },
    "tessdata": {
        "urls": [
            "https://github.com/tesseract-ocr/tessdata/archive/ced78752cc61322fb554c280d13360b35b8684e4.zip",
        ],
        "checksum": "40100cef1911bd4c1543afae3a58ca4c148bfa79eabc7e3fa5ad146ce08b103b",
        "file_checksums": {},
        "path": "tessdata",
        "zip_path": "tessdata-ced78752cc61322fb554c280d13360b35b8684e4",
    },
    "pytesseract": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:projects/tesseract/pytesseract.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:projects/tesseract/pytesseract.zip",
            "https://s3.libs.space:9000/projects/tesseract/pytesseract.zip",
        ],
        "checksum": "6eaf532ff1e3c145369c0fae7322b196dec1c9f97de619920347289053a2db09",
        "file_checksums": {},
        "path": "pytesseract",
    },
}

class OCRMonitorPlugin(Plugins.Base):
    root = None
    canvas = None
    selection_rect = None
    tkinter_thread = None
    update_queue = None
    confirm_button = None
    confirm_text = None

    tesseract_module = None

    plugin_dir = Path(Path.cwd() / "Plugins" / "ocr_monitor_plugin")
    download_state = {"is_downloading": False}

    # Region selection state
    drag_data = {"x": 0, "y": 0, "action": None}
    resize_handles = []
    handle_size = 8

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

    def is_text_changed(self, new_text, old_text):
        # overall distance check
        threshold = self.get_plugin_setting("levenshtein_threshold", 5)
        if self.levenshtein_distance(new_text, old_text) > threshold:
            return True
        # word-level check
        word_threshold = self.get_plugin_setting("levenshtein_word_threshold", 1)
        new_words = new_text.split()
        old_words = old_text.split()
        for word in new_words:
            if old_words and min(self.levenshtein_distance(word, w) for w in old_words) > word_threshold:
                return True
        for word in old_words:
            if new_words and min(self.levenshtein_distance(word, w) for w in new_words) > word_threshold:
                return True
        return False

    def handle_text(self, text):
        now = time.time()
        # detect change
        if self.prev_text is None or self.is_text_changed(text, self.prev_text):
            self.prev_text = text
            self.text_stable_since = now
            self.tts_played_for = None
            return
        stability_time = self.get_plugin_setting("stability_time", 0)
        # if text stable long enough, play TTS once
        if now - (self.text_stable_since or now) >= stability_time:
            if self.tts_played_for != text:
                self.tts_played_for = text
                original_text = None
                # translate text if translation is enabled
                if self.get_plugin_setting("text_translation_enabled"):
                    src_lang = self.get_plugin_setting("language_source")
                    target_lang = self.get_plugin_setting("language_target")
                    text = self.run_text_translate(text, src_lang, target_lang)
                if self.tts_played_for != text:
                    original_text = self.tts_played_for
                self.send_text_result(text, original_text)
                self.run_tts(text)

    def tkinter_thread_func(self):
        def check_queue():
            try:
                action = self.update_queue.get_nowait()
                if action == "close":
                    # Quit and schedule destroy in GUI thread
                    self.root.quit()
                    self.root.after_idle(lambda: self.root.destroy())
                    return
                if action == "update_region":
                    self.update_selection_rect()
            except queue.Empty:
                pass
            self.root.after(50, lambda: check_queue())

        def on_mouse_press(event):
            # Convert canvas coordinates to screen coordinates
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)

            # Check if clicking on a resize handle
            for i, handle in enumerate(self.resize_handles):
                x1, y1, x2, y2 = self.canvas.coords(handle)
                if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                    self.drag_data = {"x": canvas_x, "y": canvas_y, "action": "resize", "handle": i}
                    return

            # Check if clicking inside the selection rectangle
            if self.selection_rect:
                x1, y1, x2, y2 = self.canvas.coords(self.selection_rect)
                if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                    self.drag_data = {"x": canvas_x, "y": canvas_y, "action": "move"}
                    self.canvas.config(cursor="fleur")
                    return

            # Start new selection
            self.drag_data = {"x": canvas_x, "y": canvas_y, "action": "select"}

        def on_mouse_drag(event):
            if not self.drag_data["action"]:
                return

            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)

            if self.drag_data["action"] == "select":
                # Creating new selection
                if self.selection_rect:
                    self.canvas.delete(self.selection_rect)
                self.selection_rect = self.canvas.create_rectangle(
                    self.drag_data["x"], self.drag_data["y"], canvas_x, canvas_y,
                    outline="red", width=2, fill="", stipple="gray50"
                )
                self.update_resize_handles()

            elif self.drag_data["action"] == "move":
                # Moving existing selection
                if self.selection_rect:
                    dx = canvas_x - self.drag_data["x"]
                    dy = canvas_y - self.drag_data["y"]
                    self.canvas.move(self.selection_rect, dx, dy)
                    self.update_resize_handles()
                    self.drag_data["x"] = canvas_x
                    self.drag_data["y"] = canvas_y

            elif self.drag_data["action"] == "resize":
                # Resizing selection
                if self.selection_rect:
                    x1, y1, x2, y2 = self.canvas.coords(self.selection_rect)
                    handle_idx = self.drag_data["handle"]

                    # Update coordinates based on which handle is being dragged
                    if handle_idx == 0:  # Top-left
                        x1, y1 = canvas_x, canvas_y
                    elif handle_idx == 1:  # Top-right
                        x2, y1 = canvas_x, canvas_y
                    elif handle_idx == 2:  # Bottom-right
                        x2, y2 = canvas_x, canvas_y
                    elif handle_idx == 3:  # Bottom-left
                        x1, y2 = canvas_x, canvas_y

                    # Ensure minimum size
                    if abs(x2 - x1) < 20:
                        if x2 > x1:
                            x2 = x1 + 20
                        else:
                            x1 = x2 + 20
                    if abs(y2 - y1) < 20:
                        if y2 > y1:
                            y2 = y1 + 20
                        else:
                            y1 = y2 + 20

                    self.canvas.coords(self.selection_rect, x1, y1, x2, y2)
                    self.update_resize_handles()

        def on_mouse_release(event):
            if self.drag_data["action"]:
                # Save the region coordinates
                if self.selection_rect:
                    x1, y1, x2, y2 = self.canvas.coords(self.selection_rect)
                    # Convert to absolute screen coordinates
                    root_x = self.root.winfo_rootx()
                    root_y = self.root.winfo_rooty()

                    # Ensure coordinates are in correct order
                    region_x = int(min(x1, x2) + root_x)
                    region_y = int(min(y1, y2) + root_y)
                    region_width = int(abs(x2 - x1))
                    region_height = int(abs(y2 - y1))

                    self.set_plugin_setting("region_x", region_x)
                    self.set_plugin_setting("region_y", region_y)
                    self.set_plugin_setting("region_width", region_width)
                    self.set_plugin_setting("region_height", region_height)

                self.canvas.config(cursor="")
            self.drag_data = {"x": 0, "y": 0, "action": None}

        def on_mouse_motion(event):
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)

            # Change cursor based on what's under the mouse
            cursor = ""
            for handle in self.resize_handles:
                x1, y1, x2, y2 = self.canvas.coords(handle)
                if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                    cursor = "sizing"
                    break

            if not cursor and self.selection_rect:
                x1, y1, x2, y2 = self.canvas.coords(self.selection_rect)
                if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                    cursor = "fleur"

            self.canvas.config(cursor=cursor)

        def on_confirm_click(event):
            """Handle confirm button click"""
            self.update_queue.put("close")

        def on_confirm_enter(event):
            """Handle mouse enter on confirm button"""
            self.canvas.itemconfig(self.confirm_button, fill="lightgreen")

        def on_confirm_leave(event):
            """Handle mouse leave on confirm button"""
            self.canvas.itemconfig(self.confirm_button, fill="green")

        def on_key_press(event):
            if event.keysym in ("Escape", "Return", "space"):
                self.update_queue.put("close")

        # Create fullscreen transparent window
        self.root = tk.Tk()

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.configure(bg="black")
        self.root.overrideredirect(True)
        self.root.attributes('-alpha', 0.3)
        self.root.attributes('-topmost', True)
        self.root.focus_set()

        # Create canvas for drawing selection
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Add instructions
        instructions = self.canvas.create_text(
            screen_width // 2, 50,
            text="Drag to select OCR region • Drag corners to resize • Click CONFIRM or press ESC/ENTER to finish",
            fill="white", font=("Arial", 16)
        )

        # Create confirm button (initially hidden, will be positioned when selection exists)
        self.create_confirm_button()

        # Load existing region if available
        self.load_existing_region()

        # Bind events
        self.canvas.bind("<Button-1>", on_mouse_press)
        self.canvas.bind("<B1-Motion>", on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", on_mouse_release)
        self.canvas.bind("<Motion>", on_mouse_motion)
        self.root.bind("<KeyPress>", on_key_press)

        self.root.after(100, lambda: check_queue())
        self.root.mainloop()

    def create_confirm_button(self):
        """Create the confirm button (initially hidden)"""
        button_width = 80
        button_height = 30

        # Create button elements (initially positioned off-screen)
        self.confirm_button = self.canvas.create_rectangle(
            -100, -100, -100 + button_width, -100 + button_height,
            fill="green", outline="white", width=2, tags="confirm_button"
        )

        self.confirm_text = self.canvas.create_text(
            -100 + button_width//2, -100 + button_height//2,
            text="CONFIRM", fill="white", font=("Arial", 11, "bold"), tags="confirm_button"
        )

        # Bind confirm button events with safe error handling
        def on_confirm_click(event):
            self.update_queue.put("close")

        def on_confirm_enter(event):
            self.canvas.itemconfig(self.confirm_button, fill="lightgreen")

        def on_confirm_leave(event):
            self.canvas.itemconfig(self.confirm_button, fill="green")

        self.canvas.tag_bind("confirm_button", '<Button-1>', on_confirm_click)
        self.canvas.tag_bind("confirm_button", '<Enter>', on_confirm_enter)
        self.canvas.tag_bind("confirm_button", '<Leave>', on_confirm_leave)

    def update_confirm_button_position(self):
        """Position the confirm button outside the rectangle on the right-bottom side.
        If that would go off-screen, place it inside the rectangle instead."""
        if not self.selection_rect or not self.confirm_button:
            return

        x1, y1, x2, y2 = self.canvas.coords(self.selection_rect)

        button_width = 80
        button_height = 30
        padding = 10

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        # Default position: outside to the right, aligned with bottom
        button_x = max(x1, x2) + padding
        button_y = max(y1, y2) - button_height

        # Check if outside position would go off-screen
        outside_right = button_x + button_width > screen_w
        outside_bottom = button_y + button_height > screen_h
        outside_top = button_y < 0
        outside_left = button_x < 0

        # If any part would be off-screen, move it inside the rectangle
        if outside_right or outside_bottom or outside_top or outside_left:
            # Place it inside the rectangle (bottom-right corner)
            button_x = max(x1, x2) - button_width - padding
            button_y = max(y1, y2) - button_height - padding

            # Clamp inside rectangle in case it’s too small
            if button_x < min(x1, x2) + padding:
                button_x = min(x1, x2) + padding
            if button_y < min(y1, y2) + padding:
                button_y = min(y1, y2) + padding

        # Apply new coordinates
        self.canvas.coords(
            self.confirm_button,
            button_x, button_y,
            button_x + button_width, button_y + button_height
        )

        self.canvas.coords(
            self.confirm_text,
            button_x + button_width // 2, button_y + button_height // 2
        )


    def load_existing_region(self):
        """Load and display existing region selection if available"""
        region_x = self.get_plugin_setting("region_x", 100)
        region_y = self.get_plugin_setting("region_y", 100)
        region_width = self.get_plugin_setting("region_width", 200)
        region_height = self.get_plugin_setting("region_height", 100)

        # Convert to canvas coordinates
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()

        canvas_x1 = region_x - root_x
        canvas_y1 = region_y - root_y
        canvas_x2 = canvas_x1 + region_width
        canvas_y2 = canvas_y1 + region_height

        self.selection_rect = self.canvas.create_rectangle(
            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
            outline="red", width=2, fill="", stipple="gray50"
        )
        self.update_resize_handles()

    def update_resize_handles(self):
        """Update the position of resize handles"""
        # Clear existing handles
        for handle in self.resize_handles:
            self.canvas.delete(handle)
        self.resize_handles = []

        if not self.selection_rect:
            return

        x1, y1, x2, y2 = self.canvas.coords(self.selection_rect)
        half_handle = self.handle_size // 2

        # Create handles at corners
        positions = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x2, y2),  # Bottom-right
            (x1, y2),  # Bottom-left
        ]

        for x, y in positions:
            handle = self.canvas.create_rectangle(
                x - half_handle, y - half_handle,
                x + half_handle, y + half_handle,
                fill="red", outline="white", width=1
            )
            self.resize_handles.append(handle)

        # Update confirm button position when handles are updated
        self.update_confirm_button_position()

    def update_selection_rect(self):
        """Update the selection rectangle from settings"""
        if self.selection_rect:
            region_x = self.get_plugin_setting("region_x", 100)
            region_y = self.get_plugin_setting("region_y", 100)
            region_width = self.get_plugin_setting("region_width", 200)
            region_height = self.get_plugin_setting("region_height", 100)

            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()

            canvas_x1 = region_x - root_x
            canvas_y1 = region_y - root_y
            canvas_x2 = canvas_x1 + region_width
            canvas_y2 = canvas_y1 + region_height

            self.canvas.coords(self.selection_rect, canvas_x1, canvas_y1, canvas_x2, canvas_y2)
            self.update_resize_handles()

    def stop(self):
        """Stop the region selection interface"""
        # request mainloop to close
        if self.update_queue is not None:
            self.update_queue.put("close")
        # wait for thread to end
        if self.tkinter_thread is not None:
            self.tkinter_thread.join()
            self.tkinter_thread = None
        # clear references
        self.root = None
        self.canvas = None
        self.selection_rect = None
        self.confirm_button = None
        self.confirm_text = None
        self.update_queue = None
        self.resize_handles = []

    def show_region_selector(self):
        """Show the region selection interface (always restart)"""
        # Clean up any existing selector
        if getattr(self, 'tkinter_thread', None) is not None:
            self.stop()
        # Start a fresh selector
        self.update_queue = queue.Queue()
        self.tkinter_thread = threading.Thread(target=self.tkinter_thread_func, daemon=True)
        self.tkinter_thread.start()

    def init(self):
        os.makedirs(self.plugin_dir, exist_ok=True)

        # get text translation languages
        source_text_translation_languages = []
        target_text_translation_languages = []
        texttranslate_languages = texttranslate.GetInstalledLanguageNames()
        if texttranslate_languages is not None:
            source_text_translation_languages = [[lang['name'], lang['code']] for lang in texttranslate_languages]
            source_text_translation_languages.insert(0, ["Auto", "auto"])
            target_text_translation_languages = [[lang['name'], lang['code']] for lang in texttranslate_languages]

        # find default_language in target languages and get its name
        default_language = settings.SETTINGS.GetOption("trg_lang")
        if default_language in [lang['code'] for lang in texttranslate_languages]:
            for lang in texttranslate_languages:
                if lang['code'] == default_language:
                    default_language = lang['name']
                    break


        # prepare all possible plugin settings and their default values
        self.init_plugin_settings(
            {
                # General settings
                "monitoring_enabled": False,
                "frequency": 5,
                "tts_enabled": False,
                "show_selector": {"type": "button", "style": "button", "label": "Select Region"},
                "ocr_btn": {"type": "button", "style": "button", "label": "OCR Test"},

                # tesseract settings
                "use_tesseract": False,
                "tesseract_language": {"type": "select", "value": "eng", "values": ["eng"]},
                "tesseract_update_btn": {"type": "button", "style": "button", "label": "Update Languages"},


                # Display settings
                "subtitle_enabled": False,
                "subtitle_zz_info": {
                    "label": "subtitle_enabled needs the subtitle display plugin to work.",
                    "type": "label", "style": "left"},

                # Stability settings
                "stability_time": 0,
                "levenshtein_threshold": 5,
                "levenshtein_word_threshold": 1,

                # Region settings
                "region_x": 100,
                "region_y": 100,
                "region_width": 200,
                "region_height": 100,

                # Translation settings
                "text_translation_enabled": False,
                "language_source": {"type": "select_completion", "value": "Auto", "values": source_text_translation_languages},
                "language_target": {"type": "select_completion", "value": default_language, "values": target_text_translation_languages},
            },
            settings_groups={
                "General": [
                    "monitoring_enabled", "frequency", "tts_enabled", "show_selector", "ocr_btn"
                ],
                "Tesseract": [
                    "use_tesseract", "tesseract_language", "tesseract_update_btn"
                ],
                "Display": [
                    "subtitle_enabled", "subtitle_zz_info"
                ],
                "Stability": [
                    ["levenshtein_threshold", "stability_time"],
                    ["levenshtein_word_threshold"],
                ],
                "Region": [
                    ["region_x", "region_width"],
                    ["region_y", "region_height"],
                ],
                "Translation": [
                    ["language_target", "text_translation_enabled"],
                    ["language_source"],
                ]
            }
        )
        # setup OCR background thread
        self.setup_ocr_thread()
        # initialize TTS debounce state
        self.prev_text = None
        self.text_stable_since = None
        self.tts_played_for = None

        if self.get_plugin_setting("use_tesseract"):
            self.initialize_tesseract()

    def ocr_worker(self):
        """Background thread worker to perform OCR at set frequency."""
        while not self.ocr_thread_stop_event.is_set():
            frequency = self.get_plugin_setting("frequency", 5)
            # wait for frequency seconds or until stop event is set
            if self.ocr_thread_stop_event.wait(timeout=frequency):
                break
            if self.is_enabled(False) and self.get_plugin_setting("monitoring_enabled"):
                image = self.get_image_from_region()
                if image is not None:
                    result_text = self.run_ocr(image)
                    if result_text:
                        self.handle_text(result_text)

    def setup_ocr_thread(self):
        """Start or stop the OCR background thread based on enabled state."""
        if self.is_enabled(False):
            # start thread if not already running
            if not getattr(self, 'ocr_thread', None) or not self.ocr_thread.is_alive():
                self.ocr_thread_stop_event = threading.Event()
                self.ocr_thread = threading.Thread(target=self.ocr_worker, daemon=True)
                self.ocr_thread.start()
        else:
            # stop thread if running
            if getattr(self, 'ocr_thread_stop_event', None):
                self.ocr_thread_stop_event.set()
            if getattr(self, 'ocr_thread', None):
                self.ocr_thread.join()
                self.ocr_thread = None
            self.ocr_thread_stop_event = None

    def on_enable(self):
        """Called when plugin is enabled"""
        # manage OCR thread on enable
        self.setup_ocr_thread()

    def on_disable(self):
        """Called when plugin is disabled"""
        self.stop()
        # manage OCR thread on disable
        self.setup_ocr_thread()

    def on_event_received(self, message, websocket_connection=None):
        if self.is_enabled(False):
            if "type" not in message:
                return
            if message["type"] == "plugin_button_press":
                if message["value"] == "ocr_btn":
                    image = self.get_image_from_region()
                    if image is not None:
                        result_text = self.run_ocr(image)
                        if self.get_plugin_setting("text_translation_enabled", False):
                            src_lang = self.get_plugin_setting("language_source", "auto")
                            target_lang = self.get_plugin_setting("language_target", "eng_Latn")
                            result_text = self.run_text_translate(result_text, src_lang, target_lang)
                        websocket.BroadcastMessage(json.dumps({"type": "info",
                                                               "data": result_text}))
                        #self.run_tts(result_text)
                if message["value"] == "show_selector":
                    # show or stop region selector
                    self.show_region_selector()
                if message["value"] == "tesseract_update_btn":
                    if self.get_plugin_setting("use_tesseract") :
                        if self.initialize_tesseract():
                            print("Tesseract OCR initialized successfully.")
                        else:
                            print("Failed to initialize Tesseract OCR.")

    def get_image_from_region(self):
        region_x = self.get_plugin_setting("region_x")
        region_y = self.get_plugin_setting("region_y")
        region_width = self.get_plugin_setting("region_width")
        region_height = self.get_plugin_setting("region_height")

        # Capture the region
        if platform.system() == 'Windows':
            with mss.mss() as sct:
                monitor = {
                    "top": region_y,
                    "left": region_x,
                    "width": region_width,
                    "height": region_height
                }
                sct_img = sct.grab(monitor)
                arr = np.frombuffer(sct_img.raw, dtype='uint8')
                arr = arr.reshape((region_height, region_width, 4))
                img = arr[:, :, :3]             # drop alpha
                img = np.ascontiguousarray(img)
                return img
        return None

    def should_update_version_file_check(self, directory, current_version):
        # check version from VERSION file
        version_file = Path(directory / "WT_VERSION")
        if version_file.is_file():
            version = version_file.read_text().strip()
            if version != current_version:
                return True
            else:
                return False
        return True

    def write_version_file(self, directory, version):
        version_file = Path(directory / "WT_VERSION")
        version_file.write_text(version)

    def load_dependency(self, dependency_module, dependency_name="module", is_python_module=False):
        # determine version (fallback to checksum) and subdirectory for extraction
        version = dependency_module.get("version", dependency_module.get("checksum"))
        subdir = dependency_module.get("path") or dependency_module.get("zip_path")
        dest_path = Path(self.plugin_dir / subdir)
        # decide if we need to re-download
        needs_update = self.should_update_version_file_check(dest_path, version)
        if needs_update and dest_path.exists():
            # avoid deleting the plugin root itself
            if dest_path.resolve() != self.plugin_dir.resolve():
                print(f"Removing old {dependency_name} directory")
                shutil.rmtree(str(dest_path.resolve()))
            else:
                print(f"Skipping removal of plugin root directory for {dependency_name}")
        # download if missing or outdated
        if not dest_path.exists() or needs_update:
            downloader.download_extract(
                dependency_module["urls"],
                str(dest_path.resolve()),  # extract into the dependency subdirectory
                dependency_module["checksum"],
                alt_fallback=True,
                fallback_extract_func=downloader.extract_zip,
                fallback_extract_func_args=(
                    str(dest_path / os.path.basename(dependency_module["urls"][0])),  # downloaded zip now in dest_path
                    str(dest_path.resolve()),
                ),
                title=dependency_name,
                extract_format="zip"
            )
            # record version for future checks
            self.write_version_file(dest_path, version)
        # return either a loaded Python module or the path to a native dependency
        if is_python_module:
            return load_module(str(dest_path.resolve()))
        else:
            return str(dest_path.resolve())

    def initialize_tesseract(self):
        """Initialize Tesseract OCR module if available."""
        if self.tesseract_module is None and Path(self.plugin_dir / "pytesseract").is_dir():
            self.load_dependency(DEPENDENCY_LINKS["tesseract"], "Tesseract OCR", is_python_module=False)
            self.load_dependency(DEPENDENCY_LINKS["tessdata"], "Tesseract Data", is_python_module=False)
            self.tesseract_module = load_module(str(Path(self.plugin_dir / "pytesseract").resolve()))
        if self.tesseract_module is not None and Path(self.plugin_dir / DEPENDENCY_LINKS["tesseract"]["path"] / "tesseract.exe").is_file():
            self.tesseract_module.set_tesseract_cmd(str(Path(self.plugin_dir / DEPENDENCY_LINKS["tesseract"]["path"] / "tesseract.exe").resolve()))
            self.tesseract_module.set_tessdata_dir(str(Path(self.plugin_dir, DEPENDENCY_LINKS["tessdata"]["path"], DEPENDENCY_LINKS["tessdata"]["zip_path"]).resolve().as_posix()))

            languages_list = self.tesseract_module.get_languages(config='')
            self.set_plugin_setting("tesseract_language", {"type": "select", "value": self.get_plugin_setting("tesseract_language"),
                                                 "values": languages_list})
            return True
        return False

    def run_ocr(self, image):
        ocr_lang = settings.SETTINGS.GetOption("ocr_lang")
        if self.get_plugin_setting("use_tesseract"):
            if self.initialize_tesseract():
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                tesseract_lang = self.get_plugin_setting("tesseract_language")
                lines = self.tesseract_module.image_to_string(image, lang=tesseract_lang)
                return lines

        lines, image , _ = OCR.run_image_processing_from_image(image, [ocr_lang])
        return "\n".join(lines)

    def run_tts(self, text):
        if self.get_plugin_setting("tts_enabled", False):
            audio_device = settings.SETTINGS.GetOption("device_out_index")
            if audio_device is None or audio_device == -1:
                audio_device = settings.SETTINGS.GetOption("device_default_out_index")

            if tts.init():
                streamed_playback = settings.SETTINGS.GetOption("tts_streamed_playback")
                tts_wav = None
                if streamed_playback and hasattr(tts.tts, "tts_streaming"):
                    tts_wav, sample_rate = tts.tts.tts_streaming(text)

                if tts_wav is None:
                    streamed_playback = False
                    tts_wav, sample_rate = tts.tts.tts(text)

                if tts_wav is not None and not streamed_playback:
                    tts.tts.play_audio(tts_wav, audio_device)
            else:
                for plugin_inst in Plugins.plugins:
                    if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'tts'):
                        try:
                            plugin_inst.tts(text, audio_device, None, False, '')
                        except Exception as e:
                            print(f"Plugin TTS failed in Plugin {plugin_inst.__class__.__name__}:", e)
                            traceback.print_exc()

    def run_text_translate(self, text, src_lang=None, target_lang=None):
        text, from_code, to_code = texttranslate.TranslateLanguage(text, src_lang, target_lang, False, False)
        return text

    def send_text_result(self, text, original_text=None):
        """Send detected text to main application.
        original_text is only set if text was translated."""
        if self.is_enabled(False):

            # send to subtitle plugin if enabled
            if self.get_plugin_setting("subtitle_enabled", False):
                for plugin_inst in Plugins.plugins:
                    if plugin_inst.is_enabled(False) and plugin_inst.__class__.__name__ == "SubtitleDisplayPlugin" and hasattr(plugin_inst, 'update_label'):
                        try:
                            object_data = {
                                "text": text,
                            }
                            plugin_inst.update_label(object_data, True)
                        except Exception as e:
                            print(f"Plugin Subtitle failed in Plugin {plugin_inst.__class__.__name__}:", e)
                            traceback.print_exc()

            # send to ui via websocket
            result_obj = {
                "type": "llm_answer",
                "language": self.get_plugin_setting("language_target"),
                "llm_answer": text
            }
            websocket.BroadcastMessage(json.dumps(result_obj))
