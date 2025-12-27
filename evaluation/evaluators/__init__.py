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

from .humaneval_evaluator import HumanEvalEvaluator
from .gsm8k_evaluator import Gsm8kEvaluator
from .mbpp_evaluator import MBPPEvaluator
from .mmlu_evaluator import MMLUEvaluator
from .arc_c_evaluator import ARCCEvaluator
from .hellaswag_evaluator import HellaSwagEvaluator
from .math_evaluator import MATHEvaluator
from .gpqa_evaluator import GPQAEvaluator

_EVALUATORS = {
    "humaneval": HumanEvalEvaluator,
    "gsm8k": Gsm8kEvaluator,
    "mbpp": MBPPEvaluator,
    "mmlu": MMLUEvaluator,
    "arc_c": ARCCEvaluator,
    "arc_e": ARCCEvaluator,
    "hellaswag": HellaSwagEvaluator,
    "math": MATHEvaluator,
    "gpqa": GPQAEvaluator,
}


def get_evaluator(name: str, **kwargs):
    """Factory function to get evaluator by name."""
    if name not in _EVALUATORS:
        raise ValueError(
            f"Unknown evaluator: {name}. Available: {list(_EVALUATORS.keys())}"
        )
    return _EVALUATORS[name](**kwargs)