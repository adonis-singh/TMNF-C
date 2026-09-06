#!/usr/bin/env python3

import argparse
import ctypes
import ctypes.util
import time


Window = ctypes.c_ulong
Display = ctypes.c_void_p

x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
xtst = ctypes.CDLL(ctypes.util.find_library("Xtst"))

x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
x11.XOpenDisplay.restype = Display
x11.XDefaultRootWindow.argtypes = [Display]
x11.XDefaultRootWindow.restype = Window
x11.XQueryTree.argtypes = [
    Display,
    Window,
    ctypes.POINTER(Window),
    ctypes.POINTER(Window),
    ctypes.POINTER(ctypes.POINTER(Window)),
    ctypes.POINTER(ctypes.c_uint),
]
x11.XQueryTree.restype = ctypes.c_int
x11.XFetchName.argtypes = [Display, Window, ctypes.POINTER(ctypes.c_char_p)]
x11.XFetchName.restype = ctypes.c_int
x11.XFree.argtypes = [ctypes.c_void_p]
x11.XSetInputFocus.argtypes = [Display, Window, ctypes.c_int, ctypes.c_ulong]
x11.XRaiseWindow.argtypes = [Display, Window]
x11.XWarpPointer.argtypes = [
    Display,
    Window,
    Window,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_int,
    ctypes.c_int,
]
x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
x11.XStringToKeysym.restype = ctypes.c_ulong
x11.XKeysymToKeycode.argtypes = [Display, ctypes.c_ulong]
x11.XKeysymToKeycode.restype = ctypes.c_uint
x11.XFlush.argtypes = [Display]
x11.XSync.argtypes = [Display, ctypes.c_int]
xtst.XTestFakeKeyEvent.argtypes = [Display, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
xtst.XTestFakeMotionEvent.argtypes = [Display, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
xtst.XTestFakeButtonEvent.argtypes = [Display, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]


def child_windows(display: Display, window: Window) -> list[int]:
    root = Window()
    parent = Window()
    children = ctypes.POINTER(Window)()
    count = ctypes.c_uint()
    if not x11.XQueryTree(
        display,
        window,
        ctypes.byref(root),
        ctypes.byref(parent),
        ctypes.byref(children),
        ctypes.byref(count),
    ):
        raise RuntimeError("XQueryTree failed")
    result = [children[i] for i in range(count.value)]
    if children:
        x11.XFree(children)
    return result


def window_name(display: Display, window: Window) -> str:
    name = ctypes.c_char_p()
    if not x11.XFetchName(display, window, ctypes.byref(name)) or not name.value:
        return ""
    result = name.value.decode(errors="replace")
    x11.XFree(name)
    return result


def all_windows(display: Display) -> list[tuple[int, str]]:
    pending = [x11.XDefaultRootWindow(display)]
    result = []
    while pending:
        window = pending.pop()
        name = window_name(display, window)
        if name:
            result.append((window, name))
        pending.extend(child_windows(display, window))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window")
    parser.add_argument("--key", default="Return")
    parser.add_argument("--click", nargs=2, type=int, metavar=("X", "Y"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    display = x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError("XOpenDisplay failed")

    windows = all_windows(display)
    if args.list:
        for window, name in windows:
            print(f"0x{window:x} {name}")
        return

    if args.window:
        matches = [(window, name) for window, name in windows if args.window.lower() in name.lower()]
        if len(matches) != 1:
            raise RuntimeError(f"expected one window matching {args.window!r}, got {matches!r}")
        x11.XRaiseWindow(display, matches[0][0])
        x11.XSetInputFocus(display, matches[0][0], 2, 0)
        x11.XSync(display, False)

    if args.click:
        root = x11.XDefaultRootWindow(display)
        x11.XWarpPointer(display, 0, root, 0, 0, 0, 0, args.click[0], args.click[1])
        x11.XSync(display, False)
        time.sleep(0.1)
        xtst.XTestFakeButtonEvent(display, 1, True, 0)
        x11.XSync(display, False)
        time.sleep(0.1)
        xtst.XTestFakeButtonEvent(display, 1, False, 0)
        x11.XSync(display, False)
        return

    keysym = x11.XStringToKeysym(args.key.encode())
    keycode = x11.XKeysymToKeycode(display, keysym)
    if not keycode:
        raise RuntimeError(f"unknown key {args.key!r}")
    xtst.XTestFakeKeyEvent(display, keycode, True, 0)
    xtst.XTestFakeKeyEvent(display, keycode, False, 0)
    x11.XFlush(display)


if __name__ == "__main__":
    main()
