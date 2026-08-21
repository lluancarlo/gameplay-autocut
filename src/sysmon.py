"""What the machine is doing, for the strip along the bottom of the window.

A run is minutes of ffmpeg with nothing to show for it but a progress bar,
and the question that bar does not answer is whether the work is landing
on the GPU or grinding through x264 on the CPU. Three numbers answer it.

Everything here is read through ctypes rather than psutil and pynvml, for two
reasons: the numbers are wanted twice a second, which rules out shelling out to
nvidia-smi, and nothing new in the venv means nothing new in the .exe -- the
Win32 counters are in kernel32 and NVML ships with the NVIDIA driver, both
already on any machine this runs on.

That makes the CPU and memory readings Windows-only, which is where the .exe
runs. Elsewhere they come back as None and the window shows a dash rather than
a wrong number; the GPU reading works anywhere the NVIDIA driver does.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

GIB = 1 << 30

# NVML returns 0 for success and its own codes for everything else; nothing
# here distinguishes between the failures, only success from not.
NVML_SUCCESS = 0

# Enough for the longest card name NVML is likely to hand back.
NVML_NAME_LENGTH = 96

WINDOWS = sys.platform == "win32"

# Spelled out rather than taken from ctypes.wintypes, which does not import at
# all off Windows -- and the GPU half of this file works wherever the NVIDIA
# driver does, so the file has to import there even when kernel32 does not.
DWORD = ctypes.c_ulong
BOOL = ctypes.c_int


class FILETIME(ctypes.Structure):
    _fields_ = (("dwLowDateTime", DWORD), ("dwHighDateTime", DWORD))


@dataclass(frozen=True, slots=True)
class Reading:
    """One sample. Any field is None when that counter could not be read.

    The percentages are what the strip shows; the byte counts are what its
    tooltips show, since "78%" is the number you glance at and "12.4 of 16.0
    GB" is the one you go looking for.
    """

    cpu: float | None = None
    memory: float | None = None
    memory_used: int = 0
    memory_total: int = 0
    gpu: float | None = None
    vram: float | None = None
    vram_used: int = 0
    vram_total: int = 0


class _Kernel32:
    """CPU and physical memory, straight out of kernel32.

    CPU is not a number the OS keeps -- it only counts ticks -- so a percentage
    only exists between two samples. This holds the last one.
    """

    def __init__(self) -> None:
        self._dll = ctypes.WinDLL("kernel32", use_last_error=True)
        self._dll.GetSystemTimes.restype = BOOL
        self._dll.GlobalMemoryStatusEx.restype = BOOL
        # Primed here rather than left empty, so the first tick half a second
        # from now has a span to measure instead of reporting nothing.
        self._previous = self._ticks()

    def _ticks(self) -> tuple[int, int, int] | None:
        idle, kernel, user = (FILETIME() for _ in range(3))
        ok = self._dll.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        )
        if not ok:
            return None
        return tuple(
            (t.dwHighDateTime << 32) | t.dwLowDateTime for t in (idle, kernel, user)
        )

    def cpu(self) -> float | None:
        current = self._ticks()
        previous, self._previous = self._previous, current
        if current is None or previous is None:
            return None

        idle = current[0] - previous[0]
        # Kernel time already includes idle time, so the two spans below are
        # the whole of the interval and the busy share is what is left of it
        # once idle is taken out.
        total = (current[1] - previous[1]) + (current[2] - previous[2])
        if total <= 0:
            # Two samples inside one tick of the clock. No span, no percentage.
            return None
        return max(0.0, min(100.0, (total - idle) * 100.0 / total))

    def memory(self) -> tuple[float, int, int] | None:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if not self._dll.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        used = status.ullTotalPhys - status.ullAvailPhys
        return float(status.dwMemoryLoad), used, status.ullTotalPhys


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = (
        ("dwLength", DWORD),
        ("dwMemoryLoad", DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


class _Utilization(ctypes.Structure):
    _fields_ = (("gpu", ctypes.c_uint), ("memory", ctypes.c_uint))


class _MemoryInfo(ctypes.Structure):
    _fields_ = (
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    )


class _Nvml:
    """Utilisation and VRAM for the card NVENC would be encoding on.

    Device 0, not every card in the machine: the strip has room for one GPU,
    and on the machines this runs on that is the only one there is.

    NVML is cdecl on every platform, so CDLL rather than WinDLL. It is loaded
    by bare filename -- the driver puts nvml.dll in System32 and the .so on the
    loader path -- and a machine without an NVIDIA driver simply fails to load
    it, which is the same "no GPU reading" as a machine without the card.
    """

    LIBRARIES = ("nvml.dll", "libnvidia-ml.so.1")

    def __init__(self) -> None:
        self._dll = self._load()
        self._device: ctypes.c_void_p | None = None
        self.name = ""
        if self._dll is not None:
            self._open()

    @classmethod
    def _load(cls):
        for library in cls.LIBRARIES:
            try:
                return ctypes.CDLL(library)
            except OSError:
                continue
        return None

    def _open(self) -> None:
        assert self._dll is not None
        if self._dll.nvmlInit_v2() != NVML_SUCCESS:
            self._dll = None
            return
        device = ctypes.c_void_p()
        if self._dll.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(device)) != (
            NVML_SUCCESS
        ):
            # Initialised but no device 0 -- a driver with no card behind it.
            self._dll.nvmlShutdown()
            self._dll = None
            return
        self._device = device

        buffer = ctypes.create_string_buffer(NVML_NAME_LENGTH)
        if (
            self._dll.nvmlDeviceGetName(device, buffer, NVML_NAME_LENGTH)
            == NVML_SUCCESS
        ):
            self.name = buffer.value.decode("utf-8", "replace")

    @property
    def available(self) -> bool:
        return self._dll is not None and self._device is not None

    def read(self) -> tuple[float | None, float | None, int, int]:
        """Utilisation percent, VRAM percent, VRAM used, VRAM total.

        A driver that falls over mid-run (or a card reset by a hung encode)
        turns into Nones for that tick rather than an exception: the strip is
        the least important thing on screen and must never take the window
        with it.
        """
        if not self.available:
            return None, None, 0, 0

        utilization = _Utilization()
        gpu = (
            float(utilization.gpu)
            if self._dll.nvmlDeviceGetUtilizationRates(
                self._device, ctypes.byref(utilization)
            )
            == NVML_SUCCESS
            else None
        )

        info = _MemoryInfo()
        if (
            self._dll.nvmlDeviceGetMemoryInfo(self._device, ctypes.byref(info))
            != NVML_SUCCESS
            or info.total == 0
        ):
            return gpu, None, 0, 0
        return gpu, info.used * 100.0 / info.total, info.used, info.total

    def close(self) -> None:
        if self._dll is not None and self._device is not None:
            self._dll.nvmlShutdown()
        self._dll = None
        self._device = None


class Monitor:
    """Samples the three counters. One instance, sampled on a timer.

    Every call is a handful of kernel and driver calls -- microseconds, no
    processes spawned, nothing that blocks -- so the window's timer reads it
    on the UI thread directly rather than paying for a thread to do it.
    """

    def __init__(self) -> None:
        self._host = _Kernel32() if WINDOWS else None
        self._nvml = _Nvml()

    @property
    def gpu_name(self) -> str:
        return self._nvml.name

    def read(self) -> Reading:
        cpu = self._host.cpu() if self._host is not None else None

        memory = self._host.memory() if self._host is not None else None
        load, used, total = memory if memory is not None else (None, 0, 0)

        gpu, vram, vram_used, vram_total = self._nvml.read()
        return Reading(
            cpu=cpu,
            memory=load,
            memory_used=used,
            memory_total=total,
            gpu=gpu,
            vram=vram,
            vram_used=vram_used,
            vram_total=vram_total,
        )

    def close(self) -> None:
        self._nvml.close()


def gigabytes(used: int, total: int) -> str:
    """"12.4 of 16.0 GB", or empty when there is nothing to say."""
    if total <= 0:
        return ""
    return f"{used / GIB:.1f} of {total / GIB:.1f} GB"
