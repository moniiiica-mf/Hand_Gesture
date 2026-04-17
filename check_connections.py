"""
Check connections and dependencies for the Hand Gesture recognition system.
Verifies camera availability and required library installations.
"""

import sys
import importlib


def check_library(name, import_name=None):
    """Return (ok, version_or_error) for a Python library."""
    target = import_name or name
    try:
        mod = importlib.import_module(target)
        version = getattr(mod, "__version__", "installed")
        return True, version
    except ImportError as e:
        return False, str(e)


def check_camera(index=0):
    """Return (ok, message) after attempting to open a camera device."""
    try:
        import cv2
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return False, f"Camera index {index} not accessible"
        ret, _ = cap.read()
        cap.release()
        if not ret:
            return False, f"Camera index {index} opened but could not read frame"
        return True, f"Camera index {index} OK"
    except ImportError:
        return False, "OpenCV (cv2) not installed — cannot check camera"


def run_checks():
    results = []

    # Required libraries
    libraries = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
    ]

    print("=" * 50)
    print("Hand Gesture — Connection & Dependency Check")
    print("=" * 50)

    print("\n[Libraries]")
    all_libs_ok = True
    for display_name, import_name in libraries:
        ok, info = check_library(display_name, import_name)
        status = "OK" if ok else "MISSING"
        print(f"  {display_name:<22} [{status}]  {info}")
        results.append(ok)
        if not ok:
            all_libs_ok = False

    print("\n[Camera]")
    cam_ok, cam_msg = check_camera(index=0)
    status = "OK" if cam_ok else "FAIL"
    print(f"  Camera device           [{status}]  {cam_msg}")
    results.append(cam_ok)

    print("\n" + "=" * 50)
    if all(results):
        print("All checks passed. System is ready.")
    else:
        failed = results.count(False)
        print(f"{failed} check(s) failed. Please resolve the issues above.")
    print("=" * 50)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(run_checks())
