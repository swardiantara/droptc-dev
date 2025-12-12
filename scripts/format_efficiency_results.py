#!/usr/bin/env python3
"""
Scan experiments/efficiency_test and build a `results` JSON that matches the
structure expected by `visualize_efficiency_results.ipynb`.

Output structure (per model key):
{
  "model-name": {
     "cpu": {100: runtime_s, 500: runtime_s, ...},
     "cuda": {...},
     "cpu_throughput": {100: samples_per_sec, ...},
     "cuda_throughput": {...},
     "params": 0,
     "embedding_dim": "N/A"
  }
}

Notes:
- If you have a mapping of model -> params or embedding_dim, update the
  `model_metadata` variable or pass a CSV; currently params are set to 0 and
  embedding_dim to 'N/A' because this information isn't present in result.json.
"""

from pathlib import Path
import json
import argparse
from typing import Dict, Any


def normalize_device_name(name: str) -> str:
    n = name.lower()
    if n in ("cpu",):
        return "cpu"
    if "cuda" in n or "gpu" in n:
        return "gpu"
    return name


def get_scenario_name(scenario: str, model: str) -> str:
    scenario_map = {
        "droptc": "DroPTC",
        "drolove": "DroLoVe",
        "dronelog": "DroneLog",
        "neurallog": "NeuralLog",
        "transsentlog": "TransSentLog"}
    
    model_map = {
        "bert-base-uncased": "BERT-base",
        "all-MiniLM-L6-v2": "MiniLM",
        "all-mpnet-base-v2": "MPNet",
        "neo-bert": "NeoBERT",
        "modern-bert": "ModernBERT"
    }
    if scenario == 'droptc':
        return f"{scenario_map.get(scenario, scenario)}-{model_map.get(model, model)}"
    else:
        return scenario_map.get(scenario, scenario)


def build_results(base_path: Path) -> Dict[str, Any]:
    results = {}

    # Placeholder for manual metadata - extend this dict if you know params/emb dim
    model_metadata: Dict[str, Dict[str, Any]] = {
        "all-MiniLM-L6-v2": {"params": 22.7e6, "embedding_dim": 384},
        "all-mpnet-base-v2": {"params": 109e6, "embedding_dim": 768},
        "bert-base-uncased": {"params": 109e6, "embedding_dim": 768},
        "modern-bert": {"params": 150e6, "embedding_dim": 768},
        "neo-bert": {"params": 245e6, "embedding_dim": 768},
    }

    for scenario in sorted(p for p in base_path.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in scenario.iterdir() if p.is_dir()):
            model_key = model_dir.name
            scenario_name = get_scenario_name(scenario.name, model_key)
            model_entry = results.setdefault(scenario_name, {})

            # Apply any known metadata
            if model_key in model_metadata:
                model_entry.update(model_metadata[model_key])
            else:
                model_entry.setdefault("params", 0)
                model_entry.setdefault("embedding_dim", 0)

            for device_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                device_name_raw = device_dir.name
                device = normalize_device_name(device_name_raw)

                runtimes = model_entry.setdefault(device, {})
                throughput_key = f"{device}_throughput"
                throughputs = model_entry.setdefault(throughput_key, {})

                for sample_dir in sorted(p for p in device_dir.iterdir() if p.is_dir()):
                    result_file = sample_dir / "result.json"
                    if not result_file.exists():
                        continue
                    try:
                        data = json.loads(result_file.read_text(encoding="utf-8"))
                    except Exception:
                        # skip malformed
                        continue

                    sample_size = int(data.get("sample_size", sample_dir.name))
                    inference_time = data.get("inference_time_s")
                    if inference_time is None:
                        continue

                    # store runtime
                    runtimes[sample_size] = float(inference_time)

                    # compute throughput samples/sec
                    try:
                        throughput = float(sample_size) / float(inference_time) if inference_time > 0 else 0.0
                    except Exception:
                        throughput = 0.0
                    throughputs[sample_size] = throughput

    # Convert keys to strings for JSON compatibility (optional)
    # but keep numbers as numbers where possible; we'll write as-is
    return results


def main():
    parser = argparse.ArgumentParser(description="Format efficiency test results into a results.json")
    parser.add_argument("--base", default="experiments/efficiency_test", help="Path to efficiency_test folder")
    parser.add_argument("--out", default="experiments/efficiency_test/results_formatted.json", help="Output JSON path")
    args = parser.parse_args()

    base = Path(args.base)
    out = Path(args.out)

    if not base.exists():
        print(f"Base path not found: {base}")
        raise SystemExit(1)

    results = build_results(base)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote formatted results to: {out}")


if __name__ == "__main__":
    main()
