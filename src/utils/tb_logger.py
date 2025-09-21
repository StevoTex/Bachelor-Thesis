# File: src/utils/tb_logger.py

from __future__ import annotations

import os
import json
import threading
from typing import Any, Optional

import numpy as np
import tensorflow as tf


class TBLogger:
    """Thread-safe tf.summary wrapper for TensorBoard with optional hparams logging."""

    def __init__(self, log_dir: str, hparams: Optional[dict] = None):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self._lock = threading.Lock()

        # Optional: log hyperparameters once at step 0
        if hparams:
            with self._lock, self.writer.as_default():
                # Pretty JSON dump for readability
                tf.summary.text("HParams", json.dumps(hparams, indent=2, ensure_ascii=False), step=0)
                # Also log numeric/bool hparams as scalars for filtering
                for k, v in hparams.items():
                    if isinstance(v, (int, float, bool)):
                        tf.summary.scalar(name=f"HP/{k}", data=float(v), step=0)
                self.writer.flush()

    def scalar(self, tag: str, value: Any, step: int) -> None:
        with self._lock, self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def text(self, tag: str, text: Any, step: int = 0) -> None:
        with self._lock, self.writer.as_default():
            tf.summary.text(name=tag, data=str(text), step=step)
            self.writer.flush()

    def histogram(self, tag: str, values: Any, step: int, buckets: Optional[int] = None) -> None:
        data = tf.convert_to_tensor(values)
        with self._lock, self.writer.as_default():
            if buckets is None:
                tf.summary.histogram(name=tag, data=data, step=step)
            else:
                # Some TF builds may not support 'buckets'; fall back gracefully.
                try:
                    tf.summary.histogram(name=tag, data=data, step=step, buckets=buckets)
                except TypeError:
                    tf.summary.histogram(name=tag, data=data, step=step)
            self.writer.flush()

    def flush(self) -> None:
        with self._lock:
            self.writer.flush()

    def close(self) -> None:
        with self._lock:
            self.writer.flush()
            self.writer.close()
