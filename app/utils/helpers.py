import io
import re
import os
import hashlib
import contextlib
from pathlib import Path
from typing import Tuple, List, Optional, Any, Dict
import matplotlib
matplotlib.use('Agg')  # Headless backend safe for servers and Streamlit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def format_timestamp(seconds: float) -> str:
    """Convert seconds into MM:SS or HH:MM:SS format."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    if not text:
        return ""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 checksum for a file to track changes and prevent duplicates."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_python_code(text: str) -> List[str]:
    """Extract Python code blocks from markdown text."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        pattern2 = r"```(.*?)```"
        matches = re.findall(pattern2, text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]

def execute_matplotlib_code(code_str: str) -> Tuple[Optional[Any], str]:
    """
    Safely execute Python code in an isolated scope and capture resulting Matplotlib figure.
    Returns (fig, stdout_text).
    """
    plt.close("all")
    exec_scope = {
        "plt": plt,
        "pd": pd,
        "np": np,
        "__builtins__": __builtins__
    }

    output_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buf):
            exec(code_str, exec_scope)
        fig = plt.gcf()
        stdout_text = output_buf.getvalue()
        return fig, stdout_text
    except Exception as e:
        return None, f"Execution failed: {str(e)}"
