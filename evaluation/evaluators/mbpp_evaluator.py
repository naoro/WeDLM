# Copyright 2025 Tencent wechat. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import io
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

from .base_evaluator import BaseEvaluator


class TimeOutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeOutException("Time out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def safe_execution_worker(programs: str, task_id: int, timeout: int):
    try:
        exec_globals = {}

        with swallow_io():
            with time_limit(timeout):
                exec(programs, exec_globals)

        return task_id, "pass"

    except TimeOutException:
        return task_id, "timeout"
    except AssertionError:
        return task_id, "wrong_answer"
    except Exception:
        return task_id, "failed"


class MBPPEvaluator(BaseEvaluator):
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def _process_test(
        self, test_setup_code: str, test_list: List[str], pred_code: str
    ) -> str:
        program = pred_code.rstrip() + "\n"
        if test_setup_code:
            program += test_setup_code.rstrip() + "\n"
        if test_list:
            program += "\n".join(test_list) + "\n"
        return program

    def _process_answer(self, text):
        processed = text.split("[DONE]")[0]
        processed = processed.strip()
        if processed.startswith("'") and processed.endswith("'"):
            processed = processed[1:-1]
        processed = processed.strip()
        return processed

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        futures = []
        counters = {"pass": 0, "timeout": 0, "failed": 0, "wrong_answer": 0}
        total = 0
        detailed_predictions = []

        task_id_map = {}

        with ProcessPoolExecutor() as executor:
            for i, item in enumerate(results):
                task_id = item.get("task_id")
                generation = item.get("generation")
                test_setup_code = item.get("test_setup_code", "")
                test_list = item.get("test_list", None)

                if (
                    task_id is None
                    or generation is None
                    or not isinstance(test_list, list)
                ):
                    continue

                task_id_map[task_id] = item

                total += 1
                pred = self._process_answer(generation)

                item["extracted_code"] = pred

                programs = self._process_test(test_setup_code, test_list, pred)

                futures.append(
                    executor.submit(
                        safe_execution_worker, programs, task_id, self.timeout
                    )
                )

            for fut in as_completed(futures):
                try:
                    task_id, key = fut.result()
                except Exception as e:
                    print(f"[ERROR] Task failed unexpectedly: {e}")
                    key = "failed"
                    continue

                counters[key] = counters.get(key, 0) + 1

                is_correct = key == "pass"

                if task_id in task_id_map:
                    orig_item = task_id_map[task_id]
                    orig_item["is_correct"] = is_correct
                    orig_item["execution_result"] = key

                if task_id in task_id_map:
                    info = task_id_map[task_id]
                    detail_item = {
                        "index": info.get("index"),
                        "task_id": task_id,
                        "prompt": info.get("prompt"),
                        "generation": info.get("generation"),
                        "extracted_code": info.get("extracted_code"),
                        "test_setup_code": info.get("test_setup_code"),
                        "test_list": info.get("test_list"),
                        "execution_result": key,
                        "is_correct": is_correct,
                    }
                    detailed_predictions.append(detail_item)

        score = (counters["pass"] / total * 100.0) if total > 0 else 0.0

        metrics: Dict[str, float] = {
            "score": float(f"{score:.4f}"),
            "num_total": float(total),
            "num_pass": float(counters["pass"]),
            "num_wrong_answer": float(counters["wrong_answer"]),
            "num_failed": float(counters["failed"]),
            "num_timeout": float(counters["timeout"]),
            "detailed_predictions": detailed_predictions,
        }
        return metrics
