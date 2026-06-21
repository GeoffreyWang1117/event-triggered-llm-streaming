#!/usr/bin/env python3
"""Artifact safety scanner (referenced in the paper, Sect. 7.6 Reproducibility).

Recursively flags unintended filesystem paths, usernames, hostnames/IPs, and
API-key-like secrets in the bundled source and trace files, so the released
artifact can be checked before distribution.

Usage:
    python scripts/scan_artifact.py [root_dir]      # default: repo root
Exit code 0 = clean, 1 = potential leak(s) found.
"""
import os
import re
import sys

# Patterns that should never appear in a public artifact.
PATTERNS = {
    "home/user path": re.compile(r"/home/[a-z0-9_-]+|/Users/[A-Za-z0-9_-]+"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "Google API key": re.compile(r"AIza[A-Za-z0-9_-]{30,}"),
    "Ollama-style key": re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9_-]{20,}\b"),
    "bearer token": re.compile(r"[Bb]earer\s+[A-Za-z0-9._-]{20,}"),
    "private IPv4": re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))(?:\.\d{1,3}){2,3}\b"),
}
# Allowed placeholders (not real secrets).
ALLOW = re.compile(r"your-key|example|REDACTED|xx-+|0\.0\.0\.0|127\.0\.0\.1")
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "data"}
SKIP_EXT = {".png", ".jpg", ".jpeg", ".pdf", ".pyc", ".zip", ".bin", ".pt", ".npz"}


def scan(root):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SKIP_EXT:
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if ALLOW.search(line):
                            continue
                        for label, pat in PATTERNS.items():
                            if pat.search(line):
                                rel = os.path.relpath(path, root)
                                hits.append((rel, i, label, line.strip()[:120]))
            except (IOError, OSError):
                continue
    return hits


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hits = scan(root)
    if not hits:
        print("CLEAN: no paths, hostnames, or secrets found.")
        return 0
    print(f"POTENTIAL LEAKS ({len(hits)}):")
    for rel, ln, label, snippet in hits:
        print(f"  {rel}:{ln}  [{label}]  {snippet}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
