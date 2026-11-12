"""Generate independently hash-locked dependencies using the native OS runner.

Each lock records the *installed* resolver result plus the SHA-256 of the
specific wheel pip selected on that platform. The seed and verification
environments are isolated virtual environments; Linux wheels cannot be
mistaken for Windows or macOS wheels.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = re.compile(r"[-_.]+")
VERSION = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.!+_~-]*$")


def normalized(name: str) -> str:
    value = CANONICAL.sub("-", name).lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", value):
        raise ValueError(f"Invalid distribution name: {name!r}")
    return value


def run(*args: str) -> str:
    process = subprocess.run(
        args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if process.returncode:
        raise RuntimeError(f"Command failed: {args!r}\n{process.stdout[-6000:]}")
    return process.stdout


def python_in(venv: Path) -> str:
    return str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))


def freeze_exact(python: str) -> list[tuple[str, str]]:
    raw = json.loads(run(python, "-m", "pip", "list", "--format=json"))
    versions: dict[str, str] = {}
    for package in raw:
        name = normalized(package["name"])
        version = package["version"]
        if name == "pip":
            continue
        if not VERSION.fullmatch(version) or name in versions:
            raise ValueError(f"Invalid or duplicated distribution: {package!r}")
        versions[name] = version
    if not versions:
        raise ValueError("No third-party distributions were installed")
    return sorted(versions.items())


def lock_from_wheels(pins: list[tuple[str, str]], wheels: Path) -> str:
    actual: dict[str, Path] = {}
    for wheel in wheels.glob("*.whl"):
        name = normalized(wheel.name.split("-", 1)[0])
        if name in actual:
            raise ValueError(f"Multiple wheels for {name}")
        actual[name] = wheel
    expected = {name for name, _ in pins}
    if expected != set(actual):
        raise ValueError(f"Wheel set mismatch: missing={expected - set(actual)}, extra={set(actual) - expected}")
    lines = [
        "# Generated from a fresh Python 3.12 native platform environment.",
        "# Fully pinned versions and the SHA256 of each selected wheel.",
        "# Regenerate with: python scripts/generate_portable_lock.py",
    ]
    for name, version in pins:
        checksum = hashlib.sha256(actual[name].read_bytes()).hexdigest()
        lines.append(f"{name}=={version} --hash=sha256:{checksum}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Portable locks require Python 3.12")
    if sys.platform == "win32":
        platform = "windows"
    elif sys.platform == "darwin":
        platform = "macos"
    else:
        raise SystemExit("Use the native windows-2025 or macos-15 runner")
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        seed = temp / "seed"
        run(sys.executable, "-m", "venv", str(seed))
        python = python_in(seed)
        run(python, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(ROOT / "requirements-dev.txt"))
        run(python, "-m", "pip", "check")
        pins = freeze_exact(python)
        pins_file = temp / "pins.txt"
        pins_file.write_text("".join(f"{name}=={version}\n" for name, version in pins), encoding="utf-8")
        wheels = temp / "wheels"
        wheels.mkdir()
        run(
            python, "-m", "pip", "download", "--disable-pip-version-check",
            "--no-deps", "--only-binary=:all:", "--dest", str(wheels),
            "-r", str(pins_file),
        )
        content = lock_from_wheels(pins, wheels)
        output = ROOT / f"requirements-repro-py312-{platform}.txt"
        output.write_text(content, encoding="utf-8")
        print(f"PORTABLE_LOCK_BEGIN:{platform}", flush=True)
        print(content, end="", flush=True)
        print(f"PORTABLE_LOCK_END:{platform}", flush=True)
        # This is an independent install, not a re-use of the seed environment.
        verification = temp / "verification"
        run(sys.executable, "-m", "venv", str(verification))
        verified_python = python_in(verification)
        run(verified_python, "-m", "pip", "install", "--require-hashes", "--only-binary=:all:", "-r", str(output))
        run(verified_python, "-m", "pip", "check")
        run(verified_python, "-m", "pytest", "tests", "-q")
        print(f"PORTABLE_LOCK_VERIFIED:{platform}:{len(pins)}", flush=True)


if __name__ == "__main__":
    main()
