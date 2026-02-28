import io
import os
import re
import queue
import threading
import importlib.util
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional, List
import pandas as pd
import streamlit as st

LOG_DONE = object()

class StreamQueueWriter:
    def __init__(self, log_queue: "queue.Queue[str]"):
        self.log_queue = log_queue
        self._buffer = ""

    def write(self, data):
        if not data:
            return
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.log_queue.put(line)

    def flush(self):
        if self._buffer:
            self.log_queue.put(self._buffer)
            self._buffer = ""

def _extract_epoch_info(line: str):
    if not line:
        return None
    match = re.search(r"[Ee]poch\s*(\d+)\s*/\s*(\d+)", line)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def check_train_dependencies():
    missing = []
    if importlib.util.find_spec("ultralytics") is None:
        missing.append("ultralytics")
    if importlib.util.find_spec("torch") is None:
        missing.append("torch")
    return missing

def run_yolo_training(model_path: str, data_yaml: str, train_kwargs: dict, env_vars: dict):
    buffer = io.StringIO()
    model = None
    error = None
    results = None
    save_dir = None
    with redirect_stdout(buffer), redirect_stderr(buffer):
        try:
            if env_vars:
                for key, value in env_vars.items():
                    if value:
                        os.environ[str(key)] = str(value)
            from ultralytics import YOLO

            model = YOLO(model_path)
            results = model.train(data=data_yaml, **train_kwargs)
            trainer = getattr(model, "trainer", None)
            save_dir = getattr(trainer, "save_dir", None) if trainer else None
            if save_dir is None and hasattr(results, "save_dir"):
                save_dir = getattr(results, "save_dir")
        except Exception as exc:
            error = exc
    return results, buffer.getvalue(), save_dir, error

def run_yolo_training_stream(model_path: str, data_yaml: str, train_kwargs: dict, env_vars: dict, log_queue: "queue.Queue", result_holder: dict):
    writer = StreamQueueWriter(log_queue)
    with redirect_stdout(writer), redirect_stderr(writer):
        try:
            if env_vars:
                for key, value in env_vars.items():
                    if value:
                        os.environ[str(key)] = str(value)
            from ultralytics import YOLO

            model = YOLO(model_path)
            results = model.train(data=data_yaml, **train_kwargs)
            trainer = getattr(model, "trainer", None)
            save_dir = getattr(trainer, "save_dir", None) if trainer else None
            if save_dir is None and hasattr(results, "save_dir"):
                save_dir = getattr(results, "save_dir")
            result_holder["save_dir"] = save_dir
            result_holder["results"] = results
        except Exception as exc:
            result_holder["error"] = exc
        finally:
            writer.flush()
            log_queue.put(LOG_DONE)

def collect_run_dirs(root_str: str):
    root = Path(root_str) if root_str else None
    if not root or not root.exists():
        return []
    run_dirs = []
    for result_csv in root.rglob("results.csv"):
        run_dirs.append(result_csv.parent)
    unique = sorted({p.resolve() for p in run_dirs}, key=lambda p: p.stat().st_mtime, reverse=True)
    return unique
