#!/usr/bin/env python3

"""Pre-commit hook: recompile gh-aw workflow lock files from their source markdown.

Takes one or more paths under .github/workflows/ (as passed by pre-commit for files
matching `^\\.github/workflows/[^/]+\\.(md|lock\\.yml)$`), maps each to its workflow name
(the shared basename of <name>.md and <name>.lock.yml), and runs `gh aw compile <name>`
for every unique workflow touched.

The gh-aw CLI itself is never installed globally: this script downloads the exact
release binary pinned in .github/aw/actions-lock.json for the current OS/arch into a
version-scoped cache directory outside the repo (never git-tracked), verifies its
sha256 against that release's checksums.txt, and invokes it by path. Switching pinned
versions never mutates anything in place — each version gets its own cache path — so
there's nothing to uninstall/downgrade.

This intentionally does not commit or push anything. Pre-commit's own tracked-file
modification detection is the enforcement mechanism for an existing lock file falling
out of sync; a newly created lock file (first compile of a new workflow) is called out
explicitly below since pre-commit can't detect changes to a file it doesn't know about
yet.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile

WORKFLOWS_DIR = os.path.join(".github", "workflows")
ACTIONS_LOCK_PATH = os.path.join(".github", "aw", "actions-lock.json")
GH_AW_REPO = "github/gh-aw"


def resolve_pinned_version():
    with open(ACTIONS_LOCK_PATH) as f:
        entries = json.load(f)["entries"]
    for entry in entries.values():
        if entry["repo"] == "github/gh-aw-actions/setup":
            return entry["version"]
    print(f"::error::No github/gh-aw-actions/setup entry found in {ACTIONS_LOCK_PATH}", file=sys.stderr)
    sys.exit(1)


def platform_asset_name():
    system = platform.system().lower()
    machine = platform.machine().lower()

    system_map = {"linux": "linux", "darwin": "darwin", "windows": "windows", "freebsd": "freebsd"}
    machine_map = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "i386": "386",
        "i686": "386",
        "armv7l": "arm",
    }

    if system not in system_map or machine not in machine_map:
        print(f"::error::Unsupported platform for gh-aw: {platform.system()}/{platform.machine()}", file=sys.stderr)
        sys.exit(1)

    asset = f"{system_map[system]}-{machine_map[machine]}"
    if system_map[system] == "windows":
        asset += ".exe"
    return asset


def cache_dir():
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(root, "gh-aw")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checksum(version, asset):
    with tempfile.TemporaryDirectory() as tmp:
        checksums_path = os.path.join(tmp, "checksums.txt")
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                version,
                "--repo",
                GH_AW_REPO,
                "--pattern",
                "checksums.txt",
                "--output",
                checksums_path,
                "--clobber",
            ],
            check=True,
        )
        with open(checksums_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1] == asset:
                    return parts[0]
    print(f"::error::No checksum for {asset} in {version}'s checksums.txt", file=sys.stderr)
    sys.exit(1)


def ensure_gh_aw_binary(version):
    asset = platform_asset_name()
    binary_dir = os.path.join(cache_dir(), version)
    binary_path = os.path.join(binary_dir, "gh-aw")

    if os.path.isfile(binary_path):
        return binary_path

    expected_sha256 = download_checksum(version, asset)

    # Download straight into the destination directory (as a .partial file) rather than a
    # separate system tempdir: os.replace() below must stay within one filesystem to be
    # atomic, and /tmp vs. the cache root aren't guaranteed to share one. The `finally`
    # guarantees no partial download or empty version directory survives a failed attempt —
    # nothing this function creates outlives a successful run except the verified binary
    # itself.
    os.makedirs(binary_dir, exist_ok=True)
    partial_path = binary_path + ".partial"
    try:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                version,
                "--repo",
                GH_AW_REPO,
                "--pattern",
                asset,
                "--output",
                partial_path,
                "--clobber",
            ],
            check=True,
        )

        actual_sha256 = sha256_of(partial_path)
        if actual_sha256 != expected_sha256:
            print(
                f"::error::Checksum mismatch for gh-aw {version} ({asset}): "
                f"expected {expected_sha256}, got {actual_sha256}",
                file=sys.stderr,
            )
            sys.exit(1)

        os.chmod(partial_path, 0o755)
        os.replace(partial_path, binary_path)
    finally:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        if not os.path.isfile(binary_path):
            try:
                os.rmdir(binary_dir)
            except OSError:
                pass  # not empty (e.g. a concurrent run already placed it) — fine either way

    return binary_path


def workflow_name_for(path):
    rel = os.path.relpath(path, WORKFLOWS_DIR)
    if rel.endswith(".lock.yml"):
        return rel[: -len(".lock.yml")]
    if rel.endswith(".md"):
        return rel[: -len(".md")]
    return None


def is_tracked(path):
    result = subprocess.run(["git", "ls-files", "--error-unmatch", path], capture_output=True)
    return result.returncode == 0


def compile_workflow(binary_path, name):
    md_path = os.path.join(WORKFLOWS_DIR, f"{name}.md")
    if not os.path.isfile(md_path):
        print(f"::error::{md_path} does not exist; cannot compile an orphan lock file for '{name}'.", file=sys.stderr)
        return False

    lock_path = os.path.join(WORKFLOWS_DIR, f"{name}.lock.yml")
    lock_existed_before = os.path.isfile(lock_path)

    result = subprocess.run([binary_path, "compile", name])
    if result.returncode != 0:
        return False

    if not lock_existed_before or not is_tracked(lock_path):
        print(f"Created {lock_path} for the first time — `git add` it and commit again.", file=sys.stderr)
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help=".github/workflows/<name>.md and/or <name>.lock.yml paths")
    args = parser.parse_args()

    names = sorted({workflow_name_for(f) for f in args.files if workflow_name_for(f)})
    if not names:
        return

    version = resolve_pinned_version()
    binary_path = ensure_gh_aw_binary(version)

    ok = True
    for name in names:
        if not compile_workflow(binary_path, name):
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
