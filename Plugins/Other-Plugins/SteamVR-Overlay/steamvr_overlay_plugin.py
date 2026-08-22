# ============================================================
# SteamVR text overlay plugin for Whispering Tiger
# Version 1.0.0
#
# Displays final and optionally intermediate speech-to-text results in a
# SteamVR overlay. The OpenVR Python binding is downloaded into this plugin's
# data directory and never installed into the application's global Python.
# ============================================================

import ctypes
import importlib
import math
import os
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

import Plugins
import downloader


PLUGIN_VERSION = "1.0.0"
OPENVR_VERSION = "2.12.1401"
OPENVR_WHEEL_URLS = [
    "https://files.pythonhosted.org/packages/46/a2/"
    "e6cb89bd1803ffe70a25c20121ad8bb533fa1f1334ce7f0f7585d35e5096/"
    "openvr-2.12.1401-py3-none-any.whl",
]
OPENVR_WHEEL_SHA256 = (
    "ef761cef3162843a8025a7812d5e244fd94ab3a992ca8f5ef0d1b0188ad56cb3"
)

PLUGIN_DATA_DIR = Path.cwd() / "Plugins" / "steamvr_overlay_plugin"
OPENVR_PACKAGE_DIR = PLUGIN_DATA_DIR / "openvr"
OPENVR_VERSION_FILE = PLUGIN_DATA_DIR / "WT_OPENVR_VERSION"


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _parse_rgb(value, default):
    try:
        parsed = ImageColor.getrgb(str(value).strip())
        if len(parsed) == 4:
            parsed = parsed[:3]
        return tuple(int(channel) for channel in parsed)
    except (TypeError, ValueError):
        return default


def _load_overlay_font(font_path, font_size):
    candidates = []
    if font_path:
        candidates.append(Path(font_path))

    if os.name == "nt":
        windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        candidates.extend(
            [
                windows_fonts / "segoeui.ttf",
                windows_fonts / "arial.ttf",
                windows_fonts / "malgun.ttf",
                windows_fonts / "msgothic.ttc",
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            ]
        )

    for candidate in candidates:
        try:
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), font_size)
        except (OSError, ValueError):
            continue

    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


def _text_width(draw, text, font, stroke_width=0):
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return max(0, box[2] - box[0])


def _wrap_text_pixels(draw, text, font, max_width, stroke_width=0):
    """Wrap text by rendered width, including languages without spaces."""
    output = []
    paragraphs = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for paragraph in paragraphs:
        if paragraph == "":
            output.append("")
            continue

        current = ""
        for character in paragraph:
            candidate = current + character
            if current and _text_width(draw, candidate, font, stroke_width) > max_width:
                output.append(current.rstrip())
                current = character.lstrip() if character.isspace() else character
            else:
                current = candidate
        if current or not output:
            output.append(current.rstrip())
    return output


def _fit_leading_ellipsis(draw, line, font, max_width, stroke_width=0):
    candidate = "... " + line.lstrip()
    while len(candidate) > 4 and _text_width(
        draw, candidate, font, stroke_width
    ) > max_width:
        candidate = "... " + candidate[5:]
    return candidate


def render_overlay_image(
    text,
    width,
    height,
    font_size,
    font_path="",
    text_color="#FFFFFF",
    background_color="#101018",
    background_opacity=0.72,
    alignment="center",
    outline=True,
):
    """Render a fixed-size RGBA texture suitable for IVROverlay.setOverlayRaw."""
    width = max(128, int(width))
    height = max(64, int(height))
    font_size = max(8, int(font_size))
    background_opacity = _clamp(float(background_opacity), 0.0, 1.0)

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    padding = max(12, font_size // 2)
    radius = max(8, font_size // 3)
    background_rgb = _parse_rgb(background_color, (16, 16, 24))
    text_rgb = _parse_rgb(text_color, (255, 255, 255))
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=radius,
        fill=background_rgb + (round(background_opacity * 255),),
    )

    font = _load_overlay_font(font_path, font_size)
    stroke_width = max(1, font_size // 24) if outline else 0
    line_spacing = max(4, font_size // 5)
    usable_width = max(1, width - padding * 2)
    lines = _wrap_text_pixels(
        draw,
        str(text),
        font,
        usable_width,
        stroke_width=stroke_width,
    )

    sample_box = draw.textbbox(
        (0, 0), "Ag", font=font, stroke_width=stroke_width
    )
    line_height = max(1, sample_box[3] - sample_box[1])
    max_lines = max(
        1,
        (height - padding * 2 + line_spacing) // (line_height + line_spacing),
    )
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        lines[0] = _fit_leading_ellipsis(
            draw, lines[0], font, usable_width, stroke_width
        )

    total_height = len(lines) * line_height + max(0, len(lines) - 1) * line_spacing
    y_position = max(padding, (height - total_height) / 2)
    alignment = alignment if alignment in ("left", "center", "right") else "center"

    for line in lines:
        line_width = _text_width(draw, line, font, stroke_width)
        if alignment == "left":
            x_position = padding
        elif alignment == "right":
            x_position = width - padding - line_width
        else:
            x_position = (width - line_width) / 2

        draw.text(
            (x_position, y_position),
            line,
            font=font,
            fill=text_rgb + (255,),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 230),
        )
        y_position += line_height + line_spacing

    return image


def make_openvr_transform(openvr_module, x, y, distance, pitch, yaw, roll):
    """Create an HMD/controller-relative OpenVR 3x4 transform matrix."""
    pitch = math.radians(float(pitch))
    yaw = math.radians(float(yaw))
    roll = math.radians(float(roll))

    sin_x, cos_x = math.sin(pitch), math.cos(pitch)
    sin_y, cos_y = math.sin(yaw), math.cos(yaw)
    sin_z, cos_z = math.sin(roll), math.cos(roll)

    # Rotation order: roll (Z), yaw (Y), then pitch (X).
    rotation = (
        (
            cos_z * cos_y,
            cos_z * sin_y * sin_x - sin_z * cos_x,
            cos_z * sin_y * cos_x + sin_z * sin_x,
        ),
        (
            sin_z * cos_y,
            sin_z * sin_y * sin_x + cos_z * cos_x,
            sin_z * sin_y * cos_x - cos_z * sin_x,
        ),
        (-sin_y, cos_y * sin_x, cos_y * cos_x),
    )

    matrix = openvr_module.HmdMatrix34_t()
    for row in range(3):
        for column in range(3):
            matrix[row][column] = rotation[row][column]
    matrix[0][3] = float(x)
    matrix[1][3] = float(y)
    matrix[2][3] = -abs(float(distance))
    return matrix


class SteamVROverlayPlugin(Plugins.Base):
    def __plugin_init__(self):
        self._state_lock = threading.RLock()
        self._dependency_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread = None

        self._openvr = None
        self._vr_system = None
        self._vr_overlay = None
        self._overlay_handle = None
        self._owns_openvr_session = False
        self._force_reconnect = False

        self._history = deque()
        self._intermediate = None
        self._preview_text = None
        self._revision = 0

        self._last_error_message = None
        self._last_error_time = 0.0

    def init(self):
        self.init_plugin_settings(
            {
                "info": {
                    "label": (
                        "Shows Whispering Tiger transcriptions inside SteamVR. "
                        "SteamVR may be started before or after Whispering Tiger. Position and style "
                        "changes are applied after saving the plugin settings."
                    ),
                    "type": "label",
                    "style": "left",
                },
                "reconnect_btn": {
                    "label": "Reconnect Overlay",
                    "type": "button",
                    "style": "primary",
                },
                "test_btn": {
                    "label": "Show Test Message",
                    "type": "button",
                    "style": "default",
                },
                "clear_btn": {
                    "label": "Clear Overlay",
                    "type": "button",
                    "style": "default",
                },
                "display_mode": {
                    "type": "select_textvalue",
                    "value": "Translation (source fallback)",
                    "values": [
                        ["Translation (source fallback)", "translation"],
                        ["Source transcript", "source"],
                        ["Source and translation", "both"],
                    ],
                },
                "show_intermediate": True,
                "history_entries": {
                    "type": "slider",
                    "min": 1,
                    "max": 6,
                    "step": 1,
                    "value": 2,
                },
                "translation_separator": {
                    "type": "textfield",
                    "value": "\\n",
                },
                "max_characters": {
                    "type": "slider",
                    "min": 100,
                    "max": 4000,
                    "step": 100,
                    "value": 1600,
                },
                "display_duration": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 60.0,
                    "step": 0.5,
                    "value": 12.0,
                },
                "fade_duration": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.25,
                    "value": 2.0,
                },
                "anchor": {
                    "type": "select_textvalue",
                    "value": "Headset",
                    "values": [
                        ["Headset", "hmd"],
                        ["Left controller", "left"],
                        ["Right controller", "right"],
                    ],
                },
                "x_offset": {
                    "type": "slider",
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "value": 0.0,
                },
                "y_offset": {
                    "type": "slider",
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "value": -0.35,
                },
                "distance": {
                    "type": "slider",
                    "min": 0.05,
                    "max": 5.0,
                    "step": 0.05,
                    "value": 1.2,
                },
                "pitch": {
                    "type": "slider",
                    "min": -180,
                    "max": 180,
                    "step": 1,
                    "value": 0,
                },
                "yaw": {
                    "type": "slider",
                    "min": -180,
                    "max": 180,
                    "step": 1,
                    "value": 0,
                },
                "roll": {
                    "type": "slider",
                    "min": -180,
                    "max": 180,
                    "step": 1,
                    "value": 0,
                },
                "width_meters": {
                    "type": "slider",
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.05,
                    "value": 1.15,
                },
                "overlay_opacity": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "value": 1.0,
                },
                "texture_width": {
                    "type": "slider",
                    "min": 512,
                    "max": 1920,
                    "step": 64,
                    "value": 1280,
                },
                "texture_height": {
                    "type": "slider",
                    "min": 128,
                    "max": 1024,
                    "step": 32,
                    "value": 384,
                },
                "font_size": {
                    "type": "slider",
                    "min": 16,
                    "max": 120,
                    "step": 2,
                    "value": 48,
                },
                "font_file": {
                    "type": "file_open",
                    "accept": ".ttf,.otf,.ttc",
                    "value": "",
                },
                "text_alignment": {
                    "type": "select_textvalue",
                    "value": "center",
                    "values": [
                        ["Left", "left"],
                        ["Center", "center"],
                        ["Right", "right"],
                    ],
                },
                "text_color": {
                    "type": "textfield",
                    "value": "#FFFFFF",
                },
                "background_color": {
                    "type": "textfield",
                    "value": "#101018",
                },
                "background_opacity": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "value": 0.72,
                },
                "text_outline": True,
            },
            settings_groups={
                "General": [
                    "info",
                    "reconnect_btn",
                    "test_btn",
                    "clear_btn",
                    "display_mode",
                    "show_intermediate",
                    "history_entries",
                    "translation_separator",
                    "max_characters",
                    "display_duration",
                    "fade_duration",
                ],
                "Position": [
                    "anchor",
                    "x_offset",
                    "y_offset",
                    "distance",
                    "pitch",
                    "yaw",
                    "roll",
                    "width_meters",
                    "overlay_opacity",
                ],
                "Appearance": [
                    "texture_width",
                    "texture_height",
                    "font_size",
                    "font_file",
                    "text_alignment",
                    "text_color",
                    "background_color",
                    "background_opacity",
                    "text_outline",
                ],
            },
        )

        with self._state_lock:
            if not self.get_plugin_setting("show_intermediate", True):
                self._intermediate = None
            self._trim_history_locked()
            self._revision += 1

        if self.is_enabled(False):
            self.start_overlay()
        else:
            self.stop_overlay()
        self._wake_event.set()

    def on_enable(self):
        self.start_overlay()

    def on_disable(self):
        self.stop_overlay()

    def on_event_received(self, message, websocket_connection=None):
        if not self.is_enabled(False) or message.get("type") != "plugin_button_press":
            return

        button = message.get("value")
        if button == "reconnect_btn":
            self.reconnect_overlay()
        elif button == "test_btn":
            self.show_test_message()
        elif button == "clear_btn":
            self.clear_overlay()

    def stt(self, text, result_obj):
        if not self.is_enabled(False):
            return
        source, translation = self._extract_texts(text, result_obj)
        if not source and not translation:
            return

        with self._state_lock:
            self._history.append((source, translation))
            self._trim_history_locked()
            self._intermediate = None
            self._preview_text = None
            self._revision += 1
        self._wake_event.set()

    def stt_intermediate(self, text, result_obj):
        if not self.is_enabled(False) or not self.get_plugin_setting(
            "show_intermediate", True
        ):
            return
        source, translation = self._extract_texts(text, result_obj)
        if not source and not translation:
            return

        with self._state_lock:
            self._intermediate = (source, translation)
            self._preview_text = None
            self._revision += 1
        self._wake_event.set()

    @staticmethod
    def _extract_texts(text, result_obj):
        result_obj = result_obj if isinstance(result_obj, dict) else {}
        source = str(result_obj.get("text") or text or "").strip()
        translation = str(result_obj.get("txt_translation") or "").strip()
        return source, translation

    def _trim_history_locked(self):
        limit = _clamp(
            _as_int(self.get_plugin_setting("history_entries", 2), 2), 1, 6
        )
        while len(self._history) > limit:
            self._history.popleft()

    def _format_entry(self, source, translation):
        mode = self.get_plugin_setting("display_mode", "translation")
        if mode == "source":
            return source or translation
        if mode == "both" and source and translation and source != translation:
            separator = str(
                self.get_plugin_setting("translation_separator", "\\n")
            ).replace("\\n", "\n")
            return source + separator + translation
        return translation or source

    def _current_display_text_locked(self):
        if self._preview_text is not None:
            output = self._preview_text
        else:
            entries = list(self._history)
            if self._intermediate is not None:
                entries.append(self._intermediate)
            output = "\n\n".join(
                formatted
                for formatted in (
                    self._format_entry(source, translation)
                    for source, translation in entries
                )
                if formatted
            )

        maximum = _clamp(
            _as_int(self.get_plugin_setting("max_characters", 1600), 1600),
            100,
            4000,
        )
        if len(output) > maximum:
            output = "..." + output[-(maximum - 3):]
        return output

    def clear_overlay(self):
        with self._state_lock:
            self._history.clear()
            self._intermediate = None
            self._preview_text = None
            self._revision += 1
        self._wake_event.set()

    def show_test_message(self):
        with self._state_lock:
            self._preview_text = (
                "Whispering Tiger SteamVR overlay\n"
                "The quick brown fox jumps over the lazy dog.\n"
                "日本語 / Deutsch / Español"
            )
            self._revision += 1
        self._wake_event.set()

    def reconnect_overlay(self):
        with self._state_lock:
            running = self._thread is not None and self._thread.is_alive()
            if self._overlay_handle is not None:
                self._force_reconnect = True
        if not running:
            self.start_overlay()
        self._wake_event.set()

    def start_overlay(self):
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                self._wake_event.set()
                return
            self._stop_event.clear()
            self._force_reconnect = False
            self._thread = threading.Thread(
                target=self._overlay_worker,
                name="WhisperingTiger-SteamVR-Overlay",
                daemon=True,
            )
            self._thread.start()

    def stop_overlay(self):
        with self._state_lock:
            worker = self._thread
            self._stop_event.set()
            self._wake_event.set()

        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=5.0)
            if worker.is_alive():
                print(
                    "[SteamVR Overlay] Worker is still stopping in the background."
                )
            else:
                with self._state_lock:
                    if self._thread is worker:
                        self._thread = None

    def _dependency_is_ready(self):
        if not (OPENVR_PACKAGE_DIR / "__init__.py").is_file():
            return False
        try:
            return (
                OPENVR_VERSION_FILE.read_text(encoding="utf-8").strip()
                == OPENVR_VERSION
            )
        except OSError:
            return False

    @staticmethod
    def _remove_stale_dependency():
        dependency_root = PLUGIN_DATA_DIR.resolve()
        targets = [OPENVR_PACKAGE_DIR]
        targets.extend(PLUGIN_DATA_DIR.glob("openvr-*.dist-info"))
        for target in targets:
            try:
                target.resolve().relative_to(dependency_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing to remove dependency outside {dependency_root}: {target}"
                ) from exc

            if target.is_symlink():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        try:
            OPENVR_VERSION_FILE.unlink()
        except FileNotFoundError:
            pass

    def _load_openvr_dependency(self):
        with self._dependency_lock:
            PLUGIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
            if not self._dependency_is_ready():
                self._remove_stale_dependency()
                print(
                    f"[SteamVR Overlay] Downloading OpenVR {OPENVR_VERSION}..."
                )
                downloaded = downloader.download_extract(
                    OPENVR_WHEEL_URLS,
                    str(PLUGIN_DATA_DIR.resolve()),
                    OPENVR_WHEEL_SHA256,
                    title=f"OpenVR {OPENVR_VERSION}",
                    extract_format="zip",
                    alt_fallback=True,
                    fallback_extract_func=downloader.extract_zip,
                    fallback_extract_func_args=(
                        str(
                            PLUGIN_DATA_DIR
                            / os.path.basename(OPENVR_WHEEL_URLS[0])
                        ),
                        str(PLUGIN_DATA_DIR.resolve()),
                    ),
                )
                if not downloaded or not (OPENVR_PACKAGE_DIR / "__init__.py").is_file():
                    raise RuntimeError("OpenVR dependency download or extraction failed.")
                OPENVR_VERSION_FILE.write_text(OPENVR_VERSION, encoding="utf-8")

            dependency_root = str(PLUGIN_DATA_DIR.resolve())
            if dependency_root not in sys.path:
                sys.path.insert(0, dependency_root)
            importlib.invalidate_caches()
            module = importlib.import_module("openvr")
            self._openvr = module
            return module

    def _overlay_worker(self):
        try:
            openvr_module = self._load_openvr_dependency()
        except Exception as exc:
            self._report_error("Could not load OpenVR", exc)
            with self._state_lock:
                self._thread = None
            return

        try:
            while not self._stop_event.is_set():
                try:
                    self._connect_overlay(openvr_module)
                    self._connected_loop(openvr_module)
                except Exception as exc:
                    self._report_error(
                        "SteamVR is unavailable; the plugin will retry", exc
                    )
                finally:
                    self._disconnect_overlay(openvr_module)

                if self._stop_event.is_set():
                    break
                self._wake_event.wait(5.0)
                self._wake_event.clear()
        finally:
            self._disconnect_overlay(openvr_module)
            with self._state_lock:
                self._thread = None

    def _connect_overlay(self, openvr_module):
        self._vr_system = openvr_module.init(openvr_module.VRApplication_Background)
        self._owns_openvr_session = True
        self._vr_overlay = openvr_module.VROverlay()

        overlay_key = f"whispering.tiger.overlay.{os.getpid()}"
        self._overlay_handle = self._vr_overlay.createOverlay(
            overlay_key, "Whispering Tiger"
        )
        self._apply_overlay_configuration(self._read_configuration())
        self._set_overlay_image(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
        self._vr_overlay.setOverlayAlpha(self._overlay_handle, 0.0)
        self._vr_overlay.showOverlay(self._overlay_handle)
        self._last_error_message = None
        print("[SteamVR Overlay] Connected to SteamVR.")

    def _disconnect_overlay(self, openvr_module):
        overlay = self._vr_overlay
        handle = self._overlay_handle
        self._overlay_handle = None
        self._vr_overlay = None
        self._vr_system = None

        if overlay is not None and handle is not None:
            try:
                overlay.hideOverlay(handle)
            except Exception:
                pass
            try:
                overlay.destroyOverlay(handle)
            except Exception:
                pass

        if self._owns_openvr_session:
            self._owns_openvr_session = False
            try:
                openvr_module.shutdown()
            except Exception:
                pass

    def _connected_loop(self, openvr_module):
        last_revision = -1
        last_configuration = None
        last_update = time.monotonic()
        last_alpha = None
        configuration_dirty = True

        while not self._stop_event.is_set():
            with self._state_lock:
                if self._force_reconnect:
                    self._force_reconnect = False
                    return
                revision = self._revision
                display_text = self._current_display_text_locked()

            if not self._poll_runtime_events(openvr_module):
                return

            if configuration_dirty:
                configuration = self._read_configuration()
                if configuration != last_configuration:
                    self._apply_overlay_configuration(configuration)
                    last_configuration = configuration
                    last_revision = -1
                configuration_dirty = False
            else:
                configuration = last_configuration

            if revision != last_revision:
                if display_text:
                    image = render_overlay_image(
                        display_text,
                        configuration["texture_width"],
                        configuration["texture_height"],
                        configuration["font_size"],
                        font_path=configuration["font_file"],
                        text_color=configuration["text_color"],
                        background_color=configuration["background_color"],
                        background_opacity=configuration["background_opacity"],
                        alignment=configuration["text_alignment"],
                        outline=configuration["text_outline"],
                    )
                    self._set_overlay_image(image)
                    alpha = configuration["overlay_opacity"]
                    self._vr_overlay.setOverlayAlpha(self._overlay_handle, alpha)
                    last_alpha = alpha
                else:
                    self._set_overlay_image(
                        Image.new("RGBA", (1, 1), (0, 0, 0, 0))
                    )
                    self._vr_overlay.setOverlayAlpha(self._overlay_handle, 0.0)
                    last_alpha = 0.0
                last_update = time.monotonic()
                last_revision = revision

            if display_text:
                alpha = self._fade_alpha(
                    time.monotonic() - last_update, configuration
                )
                if last_alpha is None or abs(alpha - last_alpha) >= 0.005:
                    self._vr_overlay.setOverlayAlpha(self._overlay_handle, alpha)
                    last_alpha = alpha

            if self._wake_event.wait(0.05):
                configuration_dirty = True
            self._wake_event.clear()

    def _poll_runtime_events(self, openvr_module):
        event = openvr_module.VREvent_t()
        while self._vr_system.pollNextEvent(event):
            if event.eventType == openvr_module.VREvent_Quit:
                acknowledge_quit = getattr(
                    self._vr_system, "acknowledgeQuit_Exiting", None
                )
                if callable(acknowledge_quit):
                    acknowledge_quit()
                return False
        return True

    @staticmethod
    def _fade_alpha(elapsed, configuration):
        opacity = configuration["overlay_opacity"]
        display_duration = configuration["display_duration"]
        fade_duration = configuration["fade_duration"]
        if display_duration <= 0 or elapsed <= display_duration:
            return opacity
        if fade_duration <= 0:
            return 0.0
        ratio = 1.0 - ((elapsed - display_duration) / fade_duration)
        return opacity * _clamp(ratio, 0.0, 1.0)

    def _read_configuration(self):
        return {
            "anchor": self.get_plugin_setting("anchor", "hmd"),
            "x_offset": _as_float(self.get_plugin_setting("x_offset", 0.0), 0.0),
            "y_offset": _as_float(
                self.get_plugin_setting("y_offset", -0.35), -0.35
            ),
            "distance": _clamp(
                _as_float(self.get_plugin_setting("distance", 1.2), 1.2),
                0.05,
                5.0,
            ),
            "pitch": _as_float(self.get_plugin_setting("pitch", 0.0), 0.0),
            "yaw": _as_float(self.get_plugin_setting("yaw", 0.0), 0.0),
            "roll": _as_float(self.get_plugin_setting("roll", 0.0), 0.0),
            "width_meters": _clamp(
                _as_float(
                    self.get_plugin_setting("width_meters", 1.15), 1.15
                ),
                0.1,
                3.0,
            ),
            "overlay_opacity": _clamp(
                _as_float(
                    self.get_plugin_setting("overlay_opacity", 1.0), 1.0
                ),
                0.0,
                1.0,
            ),
            "display_duration": max(
                0.0,
                _as_float(
                    self.get_plugin_setting("display_duration", 12.0), 12.0
                ),
            ),
            "fade_duration": max(
                0.0,
                _as_float(self.get_plugin_setting("fade_duration", 2.0), 2.0),
            ),
            "texture_width": _clamp(
                _as_int(self.get_plugin_setting("texture_width", 1280), 1280),
                128,
                1920,
            ),
            "texture_height": _clamp(
                _as_int(self.get_plugin_setting("texture_height", 384), 384),
                64,
                1024,
            ),
            "font_size": _clamp(
                _as_int(self.get_plugin_setting("font_size", 48), 48), 8, 120
            ),
            "font_file": str(self.get_plugin_setting("font_file", "") or ""),
            "text_alignment": self.get_plugin_setting(
                "text_alignment", "center"
            ),
            "text_color": str(
                self.get_plugin_setting("text_color", "#FFFFFF")
            ),
            "background_color": str(
                self.get_plugin_setting("background_color", "#101018")
            ),
            "background_opacity": _clamp(
                _as_float(
                    self.get_plugin_setting("background_opacity", 0.72), 0.72
                ),
                0.0,
                1.0,
            ),
            "text_outline": bool(
                self.get_plugin_setting("text_outline", True)
            ),
        }

    def _apply_overlay_configuration(self, configuration):
        self._vr_overlay.setOverlayWidthInMeters(
            self._overlay_handle, configuration["width_meters"]
        )

        anchor = configuration["anchor"]
        if anchor == "left":
            device_index = self._vr_system.getTrackedDeviceIndexForControllerRole(
                self._openvr.TrackedControllerRole_LeftHand
            )
        elif anchor == "right":
            device_index = self._vr_system.getTrackedDeviceIndexForControllerRole(
                self._openvr.TrackedControllerRole_RightHand
            )
        else:
            device_index = self._openvr.k_unTrackedDeviceIndex_Hmd

        invalid_index = getattr(
            self._openvr, "k_unTrackedDeviceIndexInvalid", 0xFFFFFFFF
        )
        if device_index == invalid_index or not self._vr_system.isTrackedDeviceConnected(
            device_index
        ):
            raise RuntimeError(f"Selected SteamVR anchor is not connected: {anchor}")

        transform = make_openvr_transform(
            self._openvr,
            configuration["x_offset"],
            configuration["y_offset"],
            configuration["distance"],
            configuration["pitch"],
            configuration["yaw"],
            configuration["roll"],
        )
        self._vr_overlay.setOverlayTransformTrackedDeviceRelative(
            self._overlay_handle, device_index, transform
        )

    def _set_overlay_image(self, image):
        rgba_image = image.convert("RGBA")
        width, height = rgba_image.size
        raw = rgba_image.tobytes("raw", "RGBA")
        buffer = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        self._vr_overlay.setOverlayRaw(
            self._overlay_handle, buffer, width, height, 4
        )

    def _report_error(self, context, exception):
        message = f"{context}: {exception}"
        now = time.monotonic()
        if message != self._last_error_message or now - self._last_error_time >= 30.0:
            print(f"[SteamVR Overlay] {message}")
            self._last_error_message = message
            self._last_error_time = now
