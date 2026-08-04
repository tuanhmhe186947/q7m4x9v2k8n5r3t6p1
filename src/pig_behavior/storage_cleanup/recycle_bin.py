"""Windows Recycle Bin adapter with no permanent-delete fallback."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from pathlib import Path

FO_DELETE = 3
FOF_SILENT = 0x0004
FOF_NOCONFIRMATION = 0x0010
FOF_ALLOWUNDO = 0x0040
FOF_NOERRORUI = 0x0400
FOF_WANTNUKEWARNING = 0x4000


class RecycleBinError(RuntimeError):
    """Raised when Windows cannot move an item into the Recycle Bin."""


class SHFileOperation(ctypes.Structure):
    """Windows SHFILEOPSTRUCTW layout."""

    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("wFunc", wintypes.UINT),
        ("pFrom", wintypes.LPCWSTR),
        ("pTo", wintypes.LPCWSTR),
        ("fFlags", wintypes.WORD),
        ("fAnyOperationsAborted", wintypes.BOOL),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", wintypes.LPCWSTR),
    ]


class WindowsRecycleBin:
    """Move one exact path to the Windows Recycle Bin."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise RecycleBinError("Recycle Bin operations are supported only on Windows.")
        self._operation = ctypes.windll.shell32.SHFileOperationW
        self._operation.argtypes = [ctypes.POINTER(SHFileOperation)]
        self._operation.restype = ctypes.c_int

    def move(self, path: Path) -> None:
        """Recycle a file or directory and fail closed on any shell error."""

        source = f"{path}\0\0"
        operation = SHFileOperation()
        operation.wFunc = FO_DELETE
        operation.pFrom = source
        operation.fFlags = (
            FOF_ALLOWUNDO
            | FOF_NOCONFIRMATION
            | FOF_SILENT
            | FOF_NOERRORUI
            | FOF_WANTNUKEWARNING
        )
        result = self._operation(ctypes.byref(operation))
        if result != 0:
            raise RecycleBinError(
                f"Windows Recycle Bin operation failed with code {result}."
            )
        if operation.fAnyOperationsAborted:
            raise RecycleBinError("Windows reported that the operation was aborted.")
