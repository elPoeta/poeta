"""Portales XDG (xdg-desktop-portal): atajo global para hablar y capturas de pantalla, sin tocar la pantalla a mano.

El portal es quien pide permiso al usuario (KDE muestra su propio dialogo) y quien guarda la captura: POETA nunca lee
la pantalla por su cuenta.
"""

from .dbus import PortalBus, PortalError
from .screenshot import ScreenshotConfig, register_screenshot
from .shortcuts import PortalsConfig, ShortcutService

__all__ = ["PortalBus", "PortalError", "PortalsConfig", "ScreenshotConfig", "ShortcutService", "register_screenshot"]
