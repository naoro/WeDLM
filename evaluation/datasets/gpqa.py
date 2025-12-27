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

import json
import os
from typing import List, Dict, Any, Optional

from .base_dataset import BaseDataset


class GPQADataset(BaseDataset):
    """
    GPQA (Graduate-Level Google-Proof Q&A Benchmark) dataset loader.

    Data source: JSON format.
    Loading method: Few-shot CoT (Chain-of-Thought).
    Prompt format:
    [Few-shot Examples]
    Question: ...
    A) ...
    B) ...
    C) ...
    D) ...
    Answer:

    Expected output:
    [Explanation steps]
    #### [Answer Letter]
    """

    def __init__(self):
        self.path = "data/gpqa_diamond.json"

        # Few-shot examples with high-difficulty physics and chemistry cases
        self._fewshot_prefix = (
            "Question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\n"
            "A) 10^-8 eV\n"
            "B) 10^-11 eV\n"
            "C) 10^-9 eV\n"
            "D) 10^-4 eV\n"
            "Answer:\n"
            "According to the uncertainty principle, the relationship between energy width and lifetime is given by Delta E * Delta t = hbar / 2 (or approximately hbar). Using Delta t = 10^-9 s, the energy width Delta E1 is approximately 3.3 * 10^-7 eV. For Delta t = 10^-8 s (which is 10^-11 s in the typo of the explanation, but let's assume the stricter constraint), the width is even narrower. However, to clearly resolve two states, their energy difference must be significantly greater than their natural line widths (uncertainty in energy). Since the wider state has a width of ~10^-7 eV, the energy difference must be larger than this value. Among the options, 10^-8, 10^-11, and 10^-9 eV are all smaller than the line width. Only 10^-4 eV is significantly larger than 10^-7 eV, allowing the states to be resolved.\n"
            "#### D\n\n"
            "Question: trans-cinnamaldehyde was treated with methylmagnesium bromide, forming product 1. 1 was treated with pyridinium chlorochromate, forming product 2. 2 was treated with (dimethyl(oxo)-l6-sulfaneylidene)methane in DMSO at elevated temperature, forming product 3. how many carbon atoms are there in product 3?\n"
            "A) 14\n"
            "B) 12\n"
            "C) 11\n"
            "D) 10\n"
            "Answer:\n"
            "Let's trace the carbon count through the reaction sequence:\n"
            "1. Starting material: trans-cinnamaldehyde (C9H8O) has 9 carbons.\n"
            "2. Reaction 1: Treatment with methylmagnesium bromide (Grignard reagent, CH3MgBr) adds a methyl group to the carbonyl carbon. This adds 1 carbon. Product 1 is (E)-4-phenylbut-3-en-2-ol (10 carbons).\n"
            "3. Reaction 2: Pyridinium chlorochromate (PCC) is an oxidizing agent that converts the secondary alcohol back to a ketone without affecting the carbon count. Product 2 is (E)-4-phenylbut-3-en-2-one (10 carbons).\n"
            "4. Reaction 3: Treatment with (dimethyl(oxo)-l6-sulfaneylidene)methane (Corey-Chaykovsky reagent) acts as a methylene transfer agent, converting the ketone into an epoxide or, in this conjugated system, potentially a cyclopropane via 1,4-addition followed by ring closure. Regardless of the specific mechanism (epoxidation or cyclopropanation), this reagent adds one methylene (CH2) group to the molecule. 10 carbons + 1 carbon = 11 carbons.\n"
            "Product 3 is 1-(2-phenylcyclopropyl)ethan-1-one (or the epoxide isomer), which contains 11 carbons.\n"
            "#### C\n\n"
            "Question: A spin-half particle is in a linear superposition 0.5|\\uparrow\\rangle+sqrt(3)/2|\\downarrow\\rangle of its spin-up and spin-down states. If |\\uparrow\\rangle and |\\downarrow\\rangle are the eigenstates of \\sigma{z} , then what is the expectation value up to one decimal place, of the operator 10\\sigma{z}+5\\sigma_{x} ? Here, symbols have their usual meanings\n"
            "A) 0.85\n"
            "B) 1.65\n"
            "C) -1.4\n"
            "D) -0.7\n"
            "Answer:\n"
            "The state is |psi> = 0.5|up> + (sqrt(3)/2)|down>. We need the expectation value <psi| 10*sigma_z + 5*sigma_x |psi>.\n"
            "1. Expectation of sigma_z: P(up) = |0.5|^2 = 0.25. P(down) = |sqrt(3)/2|^2 = 0.75. <sigma_z> = (+1)*0.25 + (-1)*0.75 = -0.5.\n"
            "2. Expectation of sigma_x: sigma_x flips the states. <psi|sigma_x|psi> = 2 * Real(c_up^* * c_down) = 2 * 0.5 * (sqrt(3)/2) = sqrt(3)/2 approx 0.866.\n"
            "3. Total expectation value = 10 * <sigma_z> + 5 * <sigma_x> = 10*(-0.5) + 5*(0.866) = -5 + 4.33 = -0.67.\n"
            "Rounding to one decimal place, the value is -0.7.\n"
            "#### D\n\n"
            "Question: In a parallel universe where a magnet can have an isolated North or South pole, Maxwell's equations look different. But, specifically, which of those equations are different?\n"
            "A) The ones related to the divergence and the curl of the magnetic field.\n"
            "B) The one related to the divergence of the magnetic field.\n"
            "C) The ones related to the circulation of the electric field and the divergence of the magnetic field.\n"
            "D) The one related to the circulation of the magnetic field and the flux of the electric field.\n"
            "Answer:\n"
            "If isolated magnetic poles (magnetic monopoles) exist, Maxwell's equations must be modified to include magnetic charge density (rho_m) and magnetic current density (J_m) to restore symmetry between electric and magnetic fields.\n"
            "1. Gauss's law for magnetism (Divergence of B): Currently div(B) = 0. With monopoles, it becomes div(B) ~ rho_m. So, the divergence equation changes.\n"
            "2. Faraday's law (Circulation/Curl of E): Currently curl(E) = -dB/dt. With magnetic currents, a term corresponding to the flow of magnetic charge must be added (similar to how J appears in Ampere's law). It becomes curl(E) = -dB/dt - J_m. So, the circulation of E equation changes.\n"
            "The other two equations (Gauss's law for E and Ampere-Maxwell law) involve electric charges and currents, which already exist, so their form remains structurally consistent (though symmetric terms are now justified). Therefore, the equations that change are the Divergence of B and the Circulation of E.\n"
            "#### C\n\n"
            "Question: Calculate the eigenvector of a quantum mechanical operator $\\vec{P}$ for a muon along an arbitrary direction $\\vec{n}$ lying in the x-z plane corresponding to the eigenvalue $+\\hbar/2$. Given the $X-$component, $P_x$ of the operator $P$ as $\\hbar/2$ times a 2 by 2 square matrix having elements in the first row as $(0 1)$, and that in the second row as $(1, 0)$. The $Y-$component, $P_y$ of the operator is given by the product of $\\hbar/2$ and a 2 by 2 square matrix having elements in the first row as $(0, -i)$, and that in the second row as $(i, 0)$. Finally, the $Z-$component, $P_z$ of the operator is given by the product of $\\hbar/2$  and another 2 by 2 square matrix having elements in the first row as $(1, 0)$, and that in the second row as $(0, -1)$.  What are the elements of the normalized eigenvector?\n"
            "A) (\\sqrt{2/3}\\hbar, \\sqrt{1/3}\\hbar)\n"
            "B) (\\sqrt{2/3}\\hbar \\cos(\\theta/2), \\sqrt{1/3}\\hbar \\sin (\\theta/2))\n"
            "C) (\\cos(\\theta), e^{i\\phi}\\sin (\\theta))\n"
            "D) (\\cos(\\theta/2), \\sin (\\theta/2))\n"
            "Answer:\n"
            "The operator P is the spin operator S = (hbar/2) * sigma. The direction vector n lies in the x-z plane, so n = (sin(theta), 0, cos(theta)).\n"
            "The operator along n is P_n = n . P = (hbar/2) * [sin(theta)*sigma_x + cos(theta)*sigma_z].\n"
            "Using the given matrix elements: sigma_x = ((0, 1), (1, 0)) and sigma_z = ((1, 0), (0, -1)).\n"
            "The matrix for P_n becomes (hbar/2) * ((cos(theta), sin(theta)), (sin(theta), -cos(theta))).\n"
            "We need the eigenvector (u, v) for eigenvalue +hbar/2:\n"
            "((cos(theta), sin(theta)), (sin(theta), -cos(theta))) * (u, v)^T = 1 * (u, v)^T.\n"
            "This yields the equations: u*cos(theta) + v*sin(theta) = u.\n"
            "Rearranging: v*sin(theta) = u(1 - cos(theta)).\n"
            "Using half-angle identities: 2*v*sin(theta/2)cos(theta/2) = u * 2*sin^2(theta/2).\n"
            "This simplifies to v*cos(theta/2) = u*sin(theta/2), which implies u = cos(theta/2) and v = sin(theta/2) (up to a normalization constant).\n"
            "Checking normalization: cos^2(theta/2) + sin^2(theta/2) = 1. Thus, the normalized eigenvector is (cos(theta/2), sin(theta/2)).\n"
            "#### D\n\n"
        )

    @property
    def name(self) -> str:
        return "gpqa"

    @property
    def evaluator_name(self) -> str:
        return "gpqa"

    def _build_prompt(self, question: str, options: Dict[str, str]) -> str:
        """
        Build the prompt for a single question.
        """
        options_text = ""
        for label in ["A", "B", "C", "D"]:
            if label in options:
                options_text += f"{label}) {options[label]}\n"

        current_query = f"Question: {question}\n" f"{options_text}" "Answer:\n"

        return self._fewshot_prefix + current_query

    def load(self) -> List[Dict[str, Any]]:
        """Load the GPQA JSON dataset."""
        data: List[Dict[str, Any]] = []

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"GPQA data file not found at {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to decode JSON from {self.path}")

            if not isinstance(raw_data, list):
                raise ValueError(
                    f"Expected a list in {self.path}, got {type(raw_data)}"
                )

            for idx, item in enumerate(raw_data):
                question = item.get("question", "")
                correct_answer_label = item.get("answer", "").strip().upper()

                options_map = {
                    "A": item.get("A", ""),
                    "B": item.get("B", ""),
                    "C": item.get("C", ""),
                    "D": item.get("D", ""),
                }

                if not question or not correct_answer_label:
                    continue

                if correct_answer_label not in ["A", "B", "C", "D"]:
                    raise ValueError(
                        f"Invalid answer '{correct_answer_label}' in GPQA sample at index {idx}. "
                        f"Must be A, B, C, or D."
                    )

                prompt = self._build_prompt(question, options_map)

                data.append(
                    {
                        "task_id": f"gpqa_{item.get('id', idx)}",
                        "prompt": prompt,
                        "answer": correct_answer_label,
                        "question": question,
                        "options": options_map,
                        "explanation": item.get("explanation", ""),
                        "full_answer_text": item.get("full_answer_text", ""),
                    }
                )

        if not data:
            raise ValueError(f"No valid data found in {self.path}!")

        return data

    def get_recommended_config(self) -> Optional[Dict[str, Any]]:
        """Recommended configuration for GPQA evaluation."""
        return {
            "max_new_tokens": 512,
            "temperature": 0.1,
        }