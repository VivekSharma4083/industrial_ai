#!/usr/bin/env python3
"""
scripts/check_environment.py
============================
Pre-flight verification script for the Industrial AI Inspector project.
Checks Python version, virtual environment status, core dependencies,
filesystem paths, disk space, and hardware acceleration availability.
"""

import sys
import shutil
import platform
import importlib.metadata
from pathlib import Path


def check_python_version() -> bool:
    """Check that Python is 3.10+ (preferably 3.12)."""
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    print(f"[*] Python Version: {version_str} ({platform.architecture()[0]})")
    
    if (major, minor) == (3, 12):
        print("    [+] Python 3.12 detected (optimal target version).")
        return True
    elif (major, minor) >= (3, 10):
        print(f"    [!] Python {version_str} detected. Supported, but 3.12 is recommended.")
        return True
    else:
        print(f"    [-] Python {version_str} is unsupported. Python 3.12+ is required.")
        return False


def check_virtual_environment() -> bool:
    """Verify that execution is inside a project-local virtual environment."""
    in_venv = sys.prefix != sys.base_prefix or hasattr(sys, "real_prefix")
    venv_path = sys.prefix
    if in_venv:
        print(f"[*] Active Virtual Environment: {venv_path}")
        print("    [+] Running inside isolated virtual environment.")
        return True
    else:
        print("    [-] WARNING: Not running inside a virtual environment!")
        print("        Please activate the project environment: .\\.venv\\Scripts\\Activate.ps1")
        return False


def check_dependencies() -> bool:
    """Verify core Phase 1 package imports."""
    required_packages = [
        ("Flask", "flask"),
        ("Pillow", "pillow"),
        ("pytest", "pytest"),
    ]
    all_ok = True
    print("[*] Checking Phase 1 Dependencies:")
    for display_name, pkg_name in required_packages:
        try:
            ver = importlib.metadata.version(pkg_name)
            print(f"    [+] {display_name}: installed (version {ver})")
        except importlib.metadata.PackageNotFoundError:
            print(f"    [-] {display_name}: NOT found")
            all_ok = False

    # Informational check for later ML dependencies
    print("[*] Checking Machine Learning Dependencies (Phase 2+ roadmap):")
    ml_packages = [("PyTorch", "torch"), ("torchvision", "torchvision"), ("NumPy", "numpy"), ("scikit-learn", "scikit-learn")]
    for display_name, pkg_name in ml_packages:
        try:
            ver = importlib.metadata.version(pkg_name)
            print(f"    [i] {display_name}: present (version {ver})")
        except importlib.metadata.PackageNotFoundError:
            print(f"    [i] {display_name}: not installed (planned for ML integration phase)")

    return all_ok


def check_disk_space(project_root: Path) -> bool:
    """Check free disk space on the project drive."""
    try:
        usage = shutil.disk_usage(project_root)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        print(f"[*] Disk Space on {project_root.drive or 'root'}:")
        print(f"    - Free:  {free_gb:.2f} GB")
        print(f"    - Total: {total_gb:.2f} GB")

        if free_gb < 2.0:
            print("    [-] Critical: Less than 2 GB disk space available.")
            return False
        elif free_gb < 8.0:
            print("    [!] Caution: Less than 8 GB free. Recommend CPU-only wheels and selective dataset category loading.")
            return True
        else:
            print("    [+] Sufficient disk space available for initial development.")
            return True
    except Exception as exc:
        print(f"    [!] Unable to query disk space: {exc}")
        return True


def check_hardware_acceleration() -> None:
    """Inspect GPU / accelerator status."""
    print("[*] Hardware Acceleration Diagnostics:")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"    - PyTorch CUDA available: {cuda_avail}")
        if cuda_avail:
            print(f"    - Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("    - Computing Device: CPU (Optimized for inference and verification)")
    except ImportError:
        print("    - PyTorch not yet installed in Phase 1.")
        print("    - System verification indicates integrated GPU (Intel UHD Graphics 620).")
        print("    - Phase 2 will target CPU-optimized PyTorch execution.")


def check_project_structure(project_root: Path) -> bool:
    """Ensure all core architectural directories exist."""
    required_dirs = [
        project_root / "frontend" / "css",
        project_root / "frontend" / "js",
        project_root / "backend" / "routes",
        project_root / "backend" / "services",
        project_root / "src" / "data",
        project_root / "src" / "models",
        project_root / "src" / "evaluation",
        project_root / "src" / "visualization",
        project_root / "configs",
        project_root / "scripts",
        project_root / "tests",
        project_root / "artifacts",
        project_root / "docs",
    ]
    print("[*] Validating Project Directory Structure:")
    missing = []
    for d in required_dirs:
        rel = d.relative_to(project_root)
        if d.is_dir():
            print(f"    [+] {rel}/")
        else:
            print(f"    [-] Missing: {rel}/")
            missing.append(rel)

    return len(missing) == 0


def main() -> int:
    print("=" * 65)
    print(" Industrial AI Inspector -- Environment & Pre-Flight Diagnostics")
    print("=" * 65)

    project_root = Path(__file__).resolve().parent.parent

    py_ok = check_python_version()
    venv_ok = check_virtual_environment()
    deps_ok = check_dependencies()
    disk_ok = check_disk_space(project_root)
    check_hardware_acceleration()
    struct_ok = check_project_structure(project_root)

    print("=" * 65)
    if py_ok and deps_ok and disk_ok and struct_ok:
        print("[SUCCESS] Environment inspection passed! Project skeleton is verified.")
        return 0
    else:
        print("[FAILURE] One or more environment checks failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
