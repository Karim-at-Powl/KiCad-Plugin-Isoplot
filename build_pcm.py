"""Build the KiCad Plugin and Content Manager (PCM) package.

    python build_pcm.py

Writes to ``dist/``:

- ``net-isoplot-<version>.zip``: the package. Upload it as an asset of the
  GitHub release ``v<version>``, then test it in KiCad with
  *Plugin and Content Manager > Install from File...*.
- ``submission/packages/<identifier>/``: ``metadata.json`` (with the download
  URL, SHA-256 and sizes filled in) and ``icon.png``. Copy this folder into a
  fork of https://gitlab.com/kicad/addons/metadata and open a merge request.

The version packaged is the first entry of ``versions`` in metadata.json (keep
the newest first). The submitted metadata.json replaces the one in the
metadata repo, so it must list every released version: older entries keep the
download fields they already have there.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import struct
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "dist")

# Files that go into the package's plugins/ folder (paths relative to HERE).
PLUGIN_FILES = [
    "plugin.json",
    "requirements.txt",
    "LICENSE",
    "isoplot.py",
    "live.py",
    "viewer.py",
    "distance_field.py",
    "kicad_source.py",
    "icons/icon-24.png",
    "icons/icon-48.png",
]
PACKAGE_ICON = "icons/icon-64.png"

# From the PCM schema (https://go.kicad.org/pcm/schemas/v1).
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z][-a-zA-Z0-9.]{0,98}[a-zA-Z0-9]$")
_DOWNLOAD_KEYS = ("download_url", "download_sha256", "download_size", "install_size")


def _png_size(path):
    """(width, height) of a PNG file, read from its IHDR chunk."""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path} is not a PNG")
    return struct.unpack(">II", head[16:24])


def _check(meta):
    """Stop with a message if the metadata would be rejected by the PCM."""
    with open(os.path.join(HERE, "plugin.json")) as f:
        plugin = json.load(f)
    problems = []
    if not _IDENTIFIER_RE.match(meta["identifier"]):
        problems.append(f"identifier {meta['identifier']!r} has characters the PCM rejects")
    if plugin["identifier"] != meta["identifier"]:
        problems.append("identifier differs between plugin.json and metadata.json")
    if len(meta["description"]) > 150:
        problems.append("description is longer than 150 characters")
    if not meta.get("resources", {}).get("Homepage"):
        problems.append("resources.Homepage (the source repository) is missing")
    if meta["versions"][0].get("runtime") != "ipc":
        problems.append('first version entry needs "runtime": "ipc"')
    if _png_size(os.path.join(HERE, PACKAGE_ICON)) != (64, 64):
        problems.append(f"{PACKAGE_ICON} must be 64x64")
    for rel in PLUGIN_FILES:
        if not os.path.isfile(os.path.join(HERE, rel)):
            problems.append(f"missing file {rel}")
    if problems:
        raise SystemExit("Not built:\n  " + "\n  ".join(problems))


def build():
    with open(os.path.join(HERE, "metadata.json")) as f:
        meta = json.load(f)
    _check(meta)

    version = meta["versions"][0]["version"]
    zip_name = f"net-isoplot-{version}.zip"
    zip_path = os.path.join(DIST, zip_name)
    os.makedirs(DIST, exist_ok=True)

    # The metadata.json inside the archive describes only this version and
    # carries no download fields.
    packaged = copy.deepcopy(meta)
    packaged["versions"] = [
        {k: v for k, v in meta["versions"][0].items() if k not in _DOWNLOAD_KEYS}
    ]

    install_size = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        data = (json.dumps(packaged, indent=4) + "\n").encode()
        z.writestr("metadata.json", data)
        install_size += len(data)
        for rel in PLUGIN_FILES:
            src = os.path.join(HERE, rel)
            z.write(src, "plugins/" + rel)
            install_size += os.path.getsize(src)
        z.write(os.path.join(HERE, PACKAGE_ICON), "resources/icon.png")
        install_size += os.path.getsize(os.path.join(HERE, PACKAGE_ICON))

    with open(zip_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    homepage = meta["resources"]["Homepage"].rstrip("/")
    submitted = copy.deepcopy(meta)
    submitted["versions"][0].update(
        download_url=f"{homepage}/releases/download/v{version}/{zip_name}",
        download_sha256=sha256,
        download_size=os.path.getsize(zip_path),
        install_size=install_size,
    )
    sub_dir = os.path.join(DIST, "submission", "packages", meta["identifier"])
    os.makedirs(sub_dir, exist_ok=True)
    with open(os.path.join(sub_dir, "metadata.json"), "w", newline="\n") as f:
        json.dump(submitted, f, indent=4)
        f.write("\n")
    with open(os.path.join(HERE, PACKAGE_ICON), "rb") as src, \
            open(os.path.join(sub_dir, "icon.png"), "wb") as dst:
        dst.write(src.read())

    print(f"Package:    {zip_path}")
    print(f"  sha256    {sha256}")
    print(f"  size      {submitted['versions'][0]['download_size']} bytes")
    print(f"  installed {install_size} bytes")
    print(f"Submission: {sub_dir}")
    print(f"Upload the zip to the GitHub release v{version} so this URL works:")
    print(f"  {submitted['versions'][0]['download_url']}")


if __name__ == "__main__":
    sys.exit(build())
