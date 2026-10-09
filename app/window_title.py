"""Style the native Windows title bar to match the app colors (Windows 11+).

There is no Flet API for the native caption color, so we call
DwmSetWindowAttribute (DWMWA_CAPTION_COLOR / TEXT_COLOR / BORDER_COLOR)
against the top-level window found by its title. On older Windows versions
the call is a no-op (attribute not supported), which is harmless.
"""

from __future__ import annotations

import ctypes
import threading
import time

DWMWA_CAPTION_COLOR = 35  # COLORREF 0x00BBGGRR
DWMWA_TEXT_COLOR = 36
DWMWA_BORDER_COLOR = 37


def _colorref(hex_color: str) -> int:
    """'#RRGGBB' -> Windows COLORREF (0x00BBGGRR)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b << 16) | (g << 8) | r


def apply_title_bar_theme(
    window_title: str,
    caption: str = "#ffffff",
    text: str = "#111111",
    border: str = "#e3e3e6",
    attempts: int = 40,
) -> None:
    """Best-effort: recolor the native title bar to the app palette."""
    if __import__("os").name != "nt":
        return
    values = (
        (DWMWA_CAPTION_COLOR, _colorref(caption)),
        (DWMWA_TEXT_COLOR, _colorref(text)),
        (DWMWA_BORDER_COLOR, _colorref(border)),
    )

    def worker() -> None:
        user32 = ctypes.windll.user32
        dwmapi = ctypes.windll.dwmapi
        for _ in range(attempts):
            hwnd = user32.FindWindowW(None, window_title)
            if hwnd:
                for attr, value in values:
                    v = ctypes.c_int(value)
                    # failures (older Windows / not elevated) are ignored
                    dwmapi.DwmSetWindowAttribute(
                        hwnd, ctypes.c_int(attr), ctypes.byref(v), ctypes.c_int(4)
                    )
                return
            time.sleep(0.5)

    threading.Thread(target=worker, daemon=True).start()
