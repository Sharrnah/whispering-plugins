# ============================================================
# Subtitles Display Plugin for Whispering Tiger
# V1.0.10
# See https://github.com/Sharrnah/whispering-ui
# ============================================================
#
import Plugins
import tkinter as tk
import threading
import queue
import time
from collections import deque


class SubtitleDisplayPlugin(Plugins.Base):
    root = None
    label = None
    tkinter_thread = None
    update_queue = None
    close_requested = None

    current_intermediate_transcription = None

    transcription_keep_last = True
    transcription_limit = 3
    transcription_display_time = 15
    transcriptions = []
    transcription_times = None

    def tkinter_thread_func(self):
        def update_transcription_text():
            if self.root is not None and self.is_enabled(False):
                self.update_intermediate_label_text()

            self.root.after(1000, update_transcription_text)

        def check_queue():
            if self.close_requested.get():
                self.root.quit()
                return
            try:
                text = self.update_queue.get_nowait()
                self.label.config(text=text)

                # change window height according to text.
                self.root.update_idletasks()
                self.label.update_idletasks()
                label_height = self.label.winfo_reqheight()
                window_width = int(self.get_plugin_setting("window_width"))
                window_height = label_height + 20

                current_x = self.root.winfo_x()
                current_y = self.root.winfo_y()

                square_size = int(self.get_plugin_setting("square_size"))

                self.root.geometry(f"{window_width}x{window_height}+{current_x}+{current_y}")
                self.canvas.configure(height=window_height)
                self.canvas.coords(self.canvas_label_id, window_width // 2, window_height // 2)
                self.canvas.coords(square, window_width - square_size - 5, 5, window_width - 5, 5 + square_size)

            except queue.Empty:
                pass
            self.root.after(50, check_queue)

        def on_square_click(event):
            self.square_drag_data = {"x": event.x, "y": event.y}

        def on_square_release(event):
            self.square_drag_data = None

            # Save the new position
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            self.set_plugin_setting("x_position", x)
            self.set_plugin_setting("y_position", y)

        def on_square_motion(event):
            if self.square_drag_data:
                delta_x = event.x - self.square_drag_data["x"]
                delta_y = event.y - self.square_drag_data["y"]

                x = self.root.winfo_x() + delta_x
                y = self.root.winfo_y() + delta_y

                self.root.geometry(f"+{x}+{y}")

        def on_square_enter(event):
            self.canvas.itemconfig(square, fill="red")
            self.canvas.config(cursor="fleur")

        def on_square_leave(event):
            self.canvas.itemconfig(square, fill="gray")
            self.canvas.config(cursor="")

        top_position = int(self.get_plugin_setting("top"))
        bottom_position = int(self.get_plugin_setting("bottom"))

        window_width = int(self.get_plugin_setting("window_width"))
        window_height = int(self.get_plugin_setting("window_height"))

        transparency_color = self.get_plugin_setting("transparency_color")
        opacity = float(self.get_plugin_setting("opacity"))

        font_size = int(self.get_plugin_setting("font_size"))
        font_color = self.get_plugin_setting("font_color")
        font_background_color = self.get_plugin_setting("font_background_color")
        if font_background_color == "transparent":
            font_background_color = transparency_color

        square_size = int(self.get_plugin_setting("square_size"))

        self.root = tk.Tk()
        self.close_requested = tk.BooleanVar(self.root, False)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        position = top_position if top_position != 0 else screen_height - window_height - bottom_position

        if self.get_plugin_setting("x_position", None) == -1:
            self.set_plugin_setting("x_position", (screen_width - window_width) // 2)
        if self.get_plugin_setting("y_position", None) == -1:
            self.set_plugin_setting("y_position", position)
        x_pos = int(self.get_plugin_setting("x_position", (screen_width - window_width) // 2))
        y_pos = int(self.get_plugin_setting("y_position", position))

        self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        self.root.configure(bg=transparency_color)
        self.root.overrideredirect(True)
        self.root.attributes('-alpha', opacity)
        self.root.attributes('-topmost', True)
        self.root.lift()
        if opacity < 0.0:
            self.root.attributes('-alpha', 0.8)
            self.root.wm_attributes("-transparentcolor", transparency_color)

        self.canvas = tk.Canvas(self.root, bg=transparency_color, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.label = tk.Label(self.canvas, text="", font=("Helvetica", font_size), fg=font_color, bg=font_background_color, wraplength=550,
                              highlightthickness=0, highlightbackground="black")

        square = self.canvas.create_rectangle(window_width - square_size - 5, 5, window_width - 5, square_size + 5, fill="gray", outline="", tags="square")
        self.canvas.tag_bind(square, '<Button-1>', on_square_click)
        self.canvas.tag_bind(square, '<ButtonRelease-1>', on_square_release)
        self.canvas.tag_bind(square, '<B1-Motion>', on_square_motion)
        self.canvas.tag_bind(square, '<Enter>', on_square_enter)
        self.canvas.tag_bind(square, '<Leave>', on_square_leave)

        self.canvas_label_id = self.canvas.create_window(window_width // 2, window_height // 2, window=self.label)
        self.root.after(100, check_queue)
        self.root.after(1000, update_transcription_text)
        self.root.mainloop()

    def stop(self):
        if self.root is not None:
            self.close_requested.set(True)  # Add this line
            if self.tkinter_thread is not None:
                self.tkinter_thread.join()
                self.tkinter_thread = None
            self.root = None
            self.label = None
            self.update_queue = None

    def init(self):

        # prepare all possible settings
        self.init_plugin_settings(
            {
                # General
                "transcription_keep_last": True,
                "transcription_limit": {"type": "slider", "min": 1, "max": 10, "step": 1, "value": 3},
                "transcription_display_time": {"type": "slider", "min": 0, "max": 90, "step": 1, "value": 15},
                "transcription_display_source_transcript": False,
                "reverse_order_transcriptions": True,
                "extra_intermediate_line": True,

                # Position
                "top": 30,
                "bottom": 0,
                "window_width": 800,
                "window_height": 400,
                "x_position": -1,
                "y_position": -1,

                # Styling
                "square_size": 10,
                "font_size": 20,
                "font_color": "white",
                "transparency_color": "black",
                "font_background_color": "transparent",
                "opacity": {"type": "slider", "min": -0.01, "max": 1.0, "step": 0.01, "value": -0.01},
            },
            settings_groups={
                "General": ["transcription_keep_last", "transcription_limit", "transcription_display_time", "transcription_display_source_transcript", "reverse_order_transcriptions", "extra_intermediate_line"],
                "Position": ["top", "bottom", "window_width", "window_height", "x_position", "y_position"],
                "Styling": ["square_size", "font_size", "font_color", "transparency_color", "font_background_color", "opacity"],
            }
        )

        self.transcription_keep_last = self.get_plugin_setting("transcription_keep_last")
        self.transcription_limit = int(self.get_plugin_setting("transcription_limit"))
        self.transcription_display_time = int(self.get_plugin_setting("transcription_display_time"))
        self.transcriptions = deque(maxlen=self.transcription_limit)
        self.transcription_times = {}

        if self.root is None and self.is_enabled(False):
            self.update_queue = queue.Queue()
            self.tkinter_thread = threading.Thread(target=self.tkinter_thread_func, daemon=True)
            self.tkinter_thread.start()
        else:
            self.stop()

    # Add a new method for intermediate text handling
    def update_intermediate_label_text(self):
        # Remove expired transcriptions from the queue and the timestamp dict
        current_time = time.time()
        self.transcription_keep_last = self.get_plugin_setting("transcription_keep_last")
        self.transcription_display_time = int(self.get_plugin_setting("transcription_display_time"))

        if self.transcription_keep_last and len(self.transcriptions) > 1 or not self.transcription_keep_last:
            while self.transcriptions and (current_time - self.transcription_times[self.transcriptions[0]]) > self.transcription_display_time:
                old_transcription = self.transcriptions.popleft()
                del self.transcription_times[old_transcription]

        # Create a new display list including the intermediate text
        display_list = list(self.transcriptions)
        if self.current_intermediate_transcription:
            display_list.append(self.current_intermediate_transcription)

        # Rotate transcriptions so that the last one (intermediate text) is displayed first
        if self.get_plugin_setting("reverse_order_transcriptions"):
            display_list.reverse()

        # Update the label with the new display list
        updated_text = "\n".join(display_list)
        self.update_queue.put(updated_text)

    def update_label(self, result_obj, is_final=False):
        original_text = result_obj["text"]
        if not self.get_plugin_setting("transcription_display_source_transcript") and "txt_translation" in result_obj and result_obj["txt_translation"] != "":
            translated_text = result_obj["txt_translation"]
        else:
            translated_text = original_text

        # If this is a final transcription, add to transcriptions and reset intermediate text
        if is_final or (not is_final and not self.get_plugin_setting("extra_intermediate_line")):
            self.transcriptions.append(translated_text)
            self.transcription_times[translated_text] = time.time()
            self.current_intermediate_transcription = None
        else:
            # Otherwise, update the intermediate text
            self.current_intermediate_transcription = translated_text

        self.update_intermediate_label_text()

    def stt(self, text, result_obj):
        if self.is_enabled(False) and text.strip() != "":
            self.update_label(result_obj, is_final=True)
        return

    def stt_intermediate(self, text, result_obj):
        if self.is_enabled(False) and text.strip() != "":
            self.update_label(result_obj, is_final=False)
        return

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        self.stop()
        pass
