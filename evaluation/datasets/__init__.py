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

from typing import Dict, Type

from .base_dataset import BaseDataset
from .humaneval import HumanEvalDataset
from .gsm8k import Gsm8kDataset
from .mbpp import MBPPDataset
from .mmlu import MMLUDataset
from .arc_c import ARCCDataset
from .hellaswag import HellaSwagDataset
from .math import MATHDataset
from .arc_e import ARCEDataset
from .gpqa import GPQADataset


DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    "humaneval": HumanEvalDataset,
    "gsm8k": Gsm8kDataset,
    "mbpp": MBPPDataset,
    "mmlu": MMLUDataset,
    "arc_c": ARCCDataset,
    "arc_e": ARCEDataset,
    "hellaswag": HellaSwagDataset,
    "math": MATHDataset,
    "gpqa": GPQADataset,
}


def get_dataset(name: str) -> Type[BaseDataset]:
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name]