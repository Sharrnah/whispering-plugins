# ============================================================
# OCR Monitor plugin for Whispering Tiger
# Version: 0.0.1
# This will monitor a region of the screen for text and send it to Whispering Tiger for processing.
# ============================================================
import platform
import traceback

import settings
from Models import OCR
from Models.TTS import tts

if platform.system() == 'Windows':
    import mss

import numpy as np

import Plugins
import tkinter as tk
import threading
import queue
import time

class OCRMonitorPlugin(Plugins.Base):
    root = None
    canvas = None
    selection_rect = None
    tkinter_thread = None
    update_queue = None
    confirm_button = None
    confirm_text = None

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
                self.run_tts(text)
                self.tts_played_for = text

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
        """Position the confirm button inside the lower right of the selection"""
        if not self.selection_rect or not self.confirm_button:
            return

        x1, y1, x2, y2 = self.canvas.coords(self.selection_rect)

        button_width = 80
        button_height = 30
        padding = 10

        # Position inside the lower right of selection, with padding from edges
        button_x = max(x1, x2) - button_width - padding
        button_y = max(y1, y2) - button_height - padding

        # Ensure button doesn't go outside the selection area
        if button_x < min(x1, x2) + padding:
            button_x = min(x1, x2) + padding
        if button_y < min(y1, y2) + padding:
            button_y = min(y1, y2) + padding

        # Update button position
        self.canvas.coords(
            self.confirm_button,
            button_x, button_y,
            button_x + button_width, button_y + button_height
        )

        self.canvas.coords(
            self.confirm_text,
            button_x + button_width//2, button_y + button_height//2
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
        # prepare all possible plugin settings and their default values
        self.init_plugin_settings(
            {
                # General settings
                "enabled": False,
                "frequency": 5,
                "tts_enabled": False,
                "stability_time": 0,
                "levenshtein_threshold": 5,
                "levenshtein_word_threshold": 1,

                # Region settings
                "region_x": 100,
                "region_y": 100,
                "region_width": 200,
                "region_height": 100,
                "show_selector": {"type": "button", "style": "button", "label": "Select Region"},
                "ocr_btn": {"type": "button", "style": "button", "label": "OCR"},
            },
            settings_groups={
                "General": ["enabled", "frequency", "tts_enabled", "stability_time", "levenshtein_threshold", "levenshtein_word_threshold"],
                "Region": [
                    ["region_x", "region_width", "show_selector"], # Column 1
                    ["region_y", "region_height", "ocr_btn"], # Column 2
                ],
            }
        )
        # setup OCR background thread
        self.setup_ocr_thread()
        # initialize TTS debounce state
        self.prev_text = None
        self.text_stable_since = None
        self.tts_played_for = None

    def ocr_worker(self):
        """Background thread worker to perform OCR at set frequency."""
        while not self.ocr_thread_stop_event.is_set():
            frequency = self.get_plugin_setting("frequency", 5)
            # wait for frequency seconds or until stop event is set
            if self.ocr_thread_stop_event.wait(timeout=frequency):
                break
            if self.is_enabled(False) and self.get_plugin_setting("enabled"):
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
                        self.run_tts(result_text)
                if message["value"] == "show_selector":
                    # show or stop region selector
                    self.show_region_selector()

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

    def run_ocr(self, image):
        ocr_lang = settings.SETTINGS.GetOption("ocr_lang")
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
