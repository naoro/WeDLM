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

import argparse
import json
import os
import time
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any

import ray

from wedlm import LLM, SamplingParams

from evaluation.datasets import get_dataset
from evaluation.evaluators import get_evaluator


@ray.remote(num_gpus=1)
class NanoVLLMWorker:
    """
    A Ray Actor that wraps the wedlm engine on a single GPU.
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the wedlm engine.
        """
        gpu_id = ray.get_gpu_ids()[0] if ray.get_gpu_ids() else "N/A"
        print(
            f"[Worker PID={os.getpid()}, GPU={gpu_id}] Initializing WeDLMLLM engine..."
        )

        print(f"[Worker PID={os.getpid()}] Config: {kwargs}")

        self.llm = LLM(model=model_path, tensor_parallel_size=1, **kwargs)
        print(f"[Worker PID={os.getpid()}, GPU={gpu_id}] WeDLMLLM engine initialized.")

    def warmup(self, warmup_prompt: str = "Hello, world!"):
        print(f"[Worker PID={os.getpid()}] Running warmup inference...")
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        _ = self.llm.generate([warmup_prompt], sampling_params)
        print(f"[Worker PID={os.getpid()}] Warmup complete.")

    def generate_batch(
        self, batch_data: List[Dict[str, Any]], sampling_params_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompts = [item["prompt"] for item in batch_data]
        sampling_params = SamplingParams(**sampling_params_dict)
        outputs = self.llm.generate(prompts, sampling_params)

        for i, item in enumerate(batch_data):
            generated_text = outputs[i]["text"]
            item["generation"] = generated_text

        batch_stats = {"decode_tokens": 0, "decode_forwards": 0}
        if outputs:
            stats = outputs[0].get("stats", {})
            batch_stats["decode_tokens"] = stats.get("decode_tokens", 0)
            batch_stats["decode_forwards"] = stats.get("decode_forwards", 0)

        return {"data": batch_data, "stats": batch_stats}


def create_workers_sequentially(
    model_path: str, num_gpus: int, wedlm_kwargs: dict
) -> List:
    workers = []
    base_port = wedlm_kwargs.get("nccl_port", 2333)

    for i in range(num_gpus):
        print(f"\n[INFO] Creating worker {i + 1}/{num_gpus}...")

        worker_kwargs = wedlm_kwargs.copy()
        worker_kwargs["nccl_port"] = base_port + (i * 10)

        print(
            f"[INFO] Assigning nccl_port={worker_kwargs['nccl_port']} to worker {i + 1}"
        )

        worker = NanoVLLMWorker.remote(model_path, **worker_kwargs)

        if i == 0:
            print("[INFO] Warming up first worker (this may take a while)...")
            ray.get(worker.warmup.remote())
            print("[INFO] First worker warmed up.")
        else:
            time.sleep(2)

        workers.append(worker)
        print(f"[INFO] Worker {i + 1}/{num_gpus} created successfully.")
    return workers


def parse_none_or_float(value):
    """
    Custom argparse type function to parse 'None' string as Python None,
    or convert the value to float. This allows passing None to disable
    entropy threshold for one-by-one generation mode.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() == 'none':
            return None
        return float(value)
    return float(value)


def main(args):
    print("[INFO] Initializing Ray...")

    if ray.is_initialized():
        ray.shutdown()

    unique_temp_dir = os.path.join(
        tempfile.gettempdir(), f"ray_wedlm_{os.getpid()}_{int(time.time())}"
    )
    os.makedirs(unique_temp_dir, exist_ok=True)

    print(f"[INFO] Ray Config: TempDir={unique_temp_dir}, Dashboard=Disabled")

    try:
        ray.init(
            num_gpus=args.num_gpus,
            include_dashboard=False,
            _temp_dir=unique_temp_dir,
            ignore_reinit_error=True,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize Ray: {e}")
        if os.path.exists(unique_temp_dir):
            shutil.rmtree(unique_temp_dir, ignore_errors=True)
        return

    print(f"[INFO] Ray initialized. Cluster resources: {ray.available_resources()}")

    try:
        print(f"[INFO] Loading dataset '{args.dataset_name}'...")
        try:
            DatasetClass = get_dataset(args.dataset_name)
            dataset = DatasetClass()
            all_data = dataset.load()
            print(
                f"[INFO] Loaded {len(all_data)} samples from dataset '{dataset.name}'."
            )
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            return

        recommended_config = dataset.get_recommended_config() or {}
        if "per_gpu_batch_size" in recommended_config:
            recommended_config["per_worker_batch_size"] = recommended_config.pop(
                "per_gpu_batch_size"
            )

        def get_param(cli_arg_name, config_key, default_value):
            cli_val = getattr(args, cli_arg_name, None)
            if cli_val is not None:
                return cli_val
            if config_key in recommended_config:
                print(
                    f"  - Using dataset recommendation for '{cli_arg_name}': {recommended_config[config_key]}"
                )
                return recommended_config[config_key]
            return default_value

        per_worker_batch_size = get_param(
            "per_worker_batch_size", "per_worker_batch_size", 8
        )
        max_new_tokens = get_param("max_new_tokens", "max_new_tokens", 512)
        temperature = get_param("temperature", "temperature", 0.0)

        # Determine entropy threshold display string
        entropy_threshold_display = (
            "None (one-by-one generation)" 
            if args.wedlm_entropy_threshold is None 
            else args.wedlm_entropy_threshold
        )

        print("[INFO] Runtime Config:")
        print(f"  - Batch Size (per worker): {per_worker_batch_size}")
        print(f"  - Max New Tokens: {max_new_tokens}")
        print(f"  - Temperature: {temperature}")
        print(f"  - WeDLM Window Size: {args.wedlm_window_size}")
        print(
            f"  - Thresholds: Entropy={entropy_threshold_display}, "
            f"Pos Penalty={args.wedlm_pos_penalty_factor}"
        )

        # SamplingParams now contains WeDLM-specific sampling parameters
        # When wedlm_entropy_threshold is None, it enables one-by-one generation mode
        sampling_params_dict = {
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "wedlm_entropy_threshold": args.wedlm_entropy_threshold,
            "wedlm_pos_penalty_factor": args.wedlm_pos_penalty_factor
            if args.wedlm_pos_penalty_factor is not None
            else 0.02,
            "stop": ["Question", "Problem"],
        }

        wedlm_window_size = args.wedlm_window_size

        if max_new_tokens < wedlm_window_size:
            print(
                f"[INFO] Adjusting wedlm_window_size from {wedlm_window_size} to "
                f"{max_new_tokens} because max_new_tokens ({max_new_tokens}) is smaller."
            )
            wedlm_window_size = max_new_tokens

        # Config parameters for LLM initialization
        # Note: wedlm_pos_penalty_factor is now only in SamplingParams, not Config
        wedlm_kwargs = {
            "wedlm_window_size": wedlm_window_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "enforce_eager": args.enforce_eager,
            "nccl_port": 2333,
        }

        print(f"[INFO] Engine Config Base: {wedlm_kwargs}")

        print(f"\n[INFO] Creating {args.num_gpus} WeDLMLLM Worker actors...")
        workers = create_workers_sequentially(
            model_path=args.model_path,
            num_gpus=args.num_gpus,
            wedlm_kwargs=wedlm_kwargs,
        )

        print("[INFO] Distributing tasks to workers...")
        start_time = time.time()

        chunks = [
            all_data[i : i + per_worker_batch_size]
            for i in range(0, len(all_data), per_worker_batch_size)
        ]

        tasks = []
        for i, chunk in enumerate(chunks):
            worker_id = i % args.num_gpus
            tasks.append(
                workers[worker_id].generate_batch.remote(chunk, sampling_params_dict)
            )

        results_nested = ray.get(tasks)
        total_time = time.time() - start_time
        print(f"\n[INFO] Generation finished in {total_time:.2f} seconds.")
        ray.shutdown()

        print("[INFO] Aggregating results...")

        global_decode_tokens = 0
        global_decode_forwards = 0
        final_results = []

        for result in results_nested:
            final_results.extend(result["data"])
            global_decode_tokens += result["stats"].get("decode_tokens", 0)
            global_decode_forwards += result["stats"].get("decode_forwards", 0)

        global_tokens_per_forward = (
            global_decode_tokens / global_decode_forwards
            if global_decode_forwards > 0
            else 0.0
        )

        total_sequences = len(final_results)
        global_avg_tokens_per_sequence = (
            global_decode_tokens / total_sequences if total_sequences > 0 else 0.0
        )

        print(
            f"[INFO] Global Stats: decode_tokens={global_decode_tokens}, "
            f"total_seqs={total_sequences}, "
            f"avg_tokens/seq={global_avg_tokens_per_sequence:.2f}"
        )

        final_results.sort(key=lambda x: x["task_id"])

        if len(final_results) != len(all_data):
            print(
                f"[WARNING] Count mismatch: Got {len(final_results)}, "
                f"expected {len(all_data)}."
            )

        print(f"[INFO] Running evaluator '{args.dataset_name}'...")
        try:
            evaluator = get_evaluator(args.dataset_name)
            metrics = evaluator.evaluate(final_results)
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            metrics = {"error": str(e)}

        new_metrics = {
            "global_decode_tokens": global_decode_tokens,
            "global_decode_forwards": global_decode_forwards,
            "global_tokens_per_forward": global_tokens_per_forward,
            "global_avg_tokens_per_sequence": global_avg_tokens_per_sequence,
        }

        metrics = {**new_metrics, **metrics}

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(args.output_dir, args.dataset_name, timestamp)
        os.makedirs(output_dir, exist_ok=True)

        predictions_path = os.path.join(output_dir, "predictions.jsonl")
        with open(predictions_path, "w", encoding="utf-8") as f:
            for item in final_results:
                f.write(json.dumps(item) + "\n")

        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        print(f"[INFO] Results saved to: {output_dir}")

        print("\n--- Evaluation Metrics ---")
        metrics_to_print = metrics.copy()
        if (
            "detailed_predictions" in metrics_to_print
            and len(metrics_to_print["detailed_predictions"]) > 10
        ):
            metrics_to_print["detailed_predictions"] = metrics_to_print[
                "detailed_predictions"
            ][:10]
            metrics_to_print["note"] = "Detailed predictions truncated for display."
        print(json.dumps(metrics_to_print, indent=4))
        print("--------------------------\n")

    finally:
        print("[INFO] Shutting down Ray and cleaning up...")
        if os.path.exists(unique_temp_dir):
            try:
                shutil.rmtree(unique_temp_dir)
                print(f"[INFO] Cleaned up temp dir: {unique_temp_dir}")
            except OSError as e:
                print(f"[WARNING] Failed to clean up temp dir: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation using WeDLMLLM.")

    # Required arguments
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8, help="Number of GPUs to use."
    )
    parser.add_argument(
        "--dataset-name", type=str, required=True, help="Dataset name (e.g., 'gsm8k')."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory."
    )

    # Generation parameters
    parser.add_argument(
        "--per-worker-batch-size", type=int, default=None, help="Batch size per worker."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="Max tokens to generate."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature."
    )

    # Engine configuration (Config class parameters)
    parser.add_argument(
        "--wedlm-window-size", type=int, default=16, help="WeDLM sliding window size."
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096, help="Max context length."
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0).",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=512, help="Max concurrent sequences."
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager mode (disable CUDA graphs).",
    )

    # WeDLM sampling parameters (SamplingParams class parameters)
    # Use custom type function to allow None value for one-by-one generation mode
    parser.add_argument(
        "--wedlm-entropy-threshold",
        type=parse_none_or_float,
        default=0.4,
        help="WeDLM entropy threshold for parallel decoding. Use 'None' to disable "
             "parallel decoding and enable one-by-one generation mode.",
    )
    parser.add_argument(
        "--wedlm-pos-penalty-factor",
        type=float,
        default=0.02,
        help="Position penalty factor for WeDLM decoding.",
    )

    args = parser.parse_args()
    main(args)