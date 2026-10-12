"""
NOVA — OS Action Tool (app launching)

Implements the `os_action` permission tier: Nova can launch
apps from a fixed allow-list, mapped to non-shell commands. It cannot
construct or execute an arbitrary command string. Opening a terminal or
shell is deliberately excluded here — that belongs behind the `destructive`
tier with human confirmation, not this tool.

This module is called BY the Executor node, never directly by the Director
or by content parsed from email/web sources.
"""

from __future__ import annotations
import subprocess
from datetime import datetime, timezone
from pydantic import BaseModel, Field


# Fixed allow-list: friendly name -> exact launch command (list form, no shell=True).
# Add entries here deliberately; never accept a dynamic/free-text command.
ALLOWED_APPS: dict[str, list[str]] = {
    "browser": ["xdg-open", "https://"],       # append the URL as arg [1]
    "file_manager": ["nautilus"],
    "text_editor": ["gedit"],
    "code_editor": ["code"],
    "libreoffice_writer": ["libreoffice", "--writer"],
    "libreoffice_calc": ["libreoffice", "--calc"],
    "pdf_viewer": ["xdg-open"],                  # append file path as arg [1]
}

# Apps that grant further command execution are NEVER allowed through this
# tool, regardless of what gets added to ALLOWED_APPS above. This is a hard
# safety check, not just a matter of what's currently in the dict.
FORBIDDEN_SUBSTRINGS = ("bash", "sh", "zsh", "terminal", "xterm", "gnome-terminal", "konsole")


class AppLaunchResult(BaseModel):
    app_name: str
    success: bool
    error: str | None = None
    task_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def launch_app(app_name: str, arg: str | None, task_id: str) -> AppLaunchResult:
    """
    Launch an allow-listed app. `arg` is an optional single argument (a URL
    or file path) — never a shell string, never appended to a shell command.
    """
    if app_name not in ALLOWED_APPS:
        return AppLaunchResult(
            app_name=app_name, success=False, task_id=task_id,
            error=f"'{app_name}' is not in ALLOWED_APPS. Refusing to launch.",
        )

    base_cmd = ALLOWED_APPS[app_name]

    if any(bad in " ".join(base_cmd).lower() for bad in FORBIDDEN_SUBSTRINGS):
        return AppLaunchResult(
            app_name=app_name, success=False, task_id=task_id,
            error=f"'{app_name}' resolves to a shell-granting command and is blocked at the tool level.",
        )

    cmd = base_cmd + ([arg] if arg else [])

    try:
        # shell=False is deliberate and non-negotiable: no string is ever
        # interpreted by a shell, which is what makes injection impossible here.
        subprocess.Popen(cmd, shell=False)
        return AppLaunchResult(app_name=app_name, success=True, task_id=task_id)
    except Exception as e:
        return AppLaunchResult(app_name=app_name, success=False, task_id=task_id, error=str(e))
