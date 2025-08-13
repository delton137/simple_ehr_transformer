#!/usr/bin/env python3
"""
Time-to-event evaluation via forward simulation.

For each patient, choose an index time, build the tokenized history up to that time,
simulate multiple futures for a given horizon (e.g., 1 year), and estimate the
probability of a target event (e.g., diabetes) occurring within the horizon by
the fraction of simulations that include a matching token before the horizon.

Computes metrics (AUROC, AUPRC, Brier score) and calibration (ECE, reliability plot).
"""

import os
import re
import json
import math
import pickle
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
from datetime import datetime, timedelta

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt

from config import data_config
from model import create_ethos_model
from data_processor import OMOPDataProcessor


# -----------------------------
# Utilities
# -----------------------------

TIME_LABEL_TO_MINUTES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "3h": 180,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
    "1m": 43200,
    "3m": 129600,
    "6m": 259200,
    "1y": 525600,
}


def token_minutes_from_name(token_name: str) -> Optional[int]:
    if not token_name.startswith("TIME_"):
        return None
    label = token_name.split("TIME_")[-1]
    return TIME_LABEL_TO_MINUTES.get(label)


def build_target_token_ids(
    vocab: Dict[str, int],
    event_type: Optional[str] = None,
    concept_ids: Optional[Iterable[int]] = None,
    token_regexes: Optional[Iterable[str]] = None,
) -> List[int]:
    """Construct list of target token IDs to detect in generated futures.

    - If event_type and concept_ids are provided, builds tokens like f"CONDITION_{cid}".
    - If token_regexes provided, matches any vocab tokens by regex and adds their IDs.
    """
    target_ids: List[int] = []

    if event_type and concept_ids:
        prefix_map = {
            "condition": "CONDITION_",
            "medication": "DRUG_",
            "procedure": "PROCEDURE_",
            "measurement": "MEASUREMENT_",
            "observation": "OBSERVATION_",
            "death": "DEATH",  # typically event token, not concept; kept for completeness
        }
        prefix = prefix_map.get(event_type.lower())
        if prefix:
            for cid in concept_ids:
                token_name = f"{prefix}{cid}"
                if token_name in vocab:
                    target_ids.append(vocab[token_name])

    if token_regexes:
        compiled = [re.compile(rx) for rx in token_regexes]
        for name, tid in vocab.items():
            if any(rx.search(name) for rx in compiled):
                target_ids.append(tid)

    # Deduplicate
    return sorted(set(target_ids))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute ECE and return also bin accuracies and confidences for plotting."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_acc = []
    bin_conf = []
    bin_weights = []
    for b in range(n_bins):
        idx = bin_ids == b
        if np.any(idx):
            acc = y_true[idx].mean()
            conf = y_prob[idx].mean()
            w = idx.mean()
        else:
            acc = 0.0
            conf = 0.0
            w = 0.0
        bin_acc.append(acc)
        bin_conf.append(conf)
        bin_weights.append(w)
    bin_acc = np.array(bin_acc)
    bin_conf = np.array(bin_conf)
    bin_weights = np.array(bin_weights)
    ece = float(np.sum(bin_weights * np.abs(bin_acc - bin_conf)))
    return ece, bin_acc, bin_conf


# -----------------------------
# Core evaluation
# -----------------------------

@dataclass
class EvalConfig:
    model_path: str
    data_dir: str
    output_dir: str
    device: str = "auto"
    forward_horizon_days: int = 365
    num_samples: int = 50
    max_gen_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    index_strategy: str = "end_minus_horizon"  # or "random"
    event_type: Optional[str] = None
    concept_ids: Optional[List[int]] = None
    token_regexes: Optional[List[str]] = None
    patient_limit: Optional[int] = None


class TimeToEventEvaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        # Load vocab and patient timelines
        with open(os.path.join(cfg.data_dir, "vocabulary.pkl"), "rb") as f:
            self.vocab: Dict[str, int] = pickle.load(f)
        with open(os.path.join(cfg.data_dir, "patient_timelines.pkl"), "rb") as f:
            self.patient_timelines: Dict[int, List[Dict]] = pickle.load(f)

        # Reverse vocab
        self.id_to_token = {tid: name for name, tid in self.vocab.items()}

        # Build processor with mappings consistent with training
        self.processor = OMOPDataProcessor()
        # Load quantiles; age/time mappings are deterministic
        try:
            with open(os.path.join(cfg.data_dir, "quantile_mappings.pkl"), "rb") as f:
                self.processor.quantile_mappings = pickle.load(f)
        except Exception:
            self.processor.quantile_mappings = {}
        self.processor.create_age_mappings()
        self.processor.create_time_interval_mappings()
        # Adopt the same vocabulary
        self.processor.vocab = self.vocab
        self.processor.vocab_size = len(self.vocab)

        # Load model
        checkpoint = torch.load(cfg.model_path, map_location=self.device)
        self.model = create_ethos_model(len(self.vocab))
        self.model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[index]
        self.model = self.model.to(self.device)
        self.model.eval()

        # Target tokens
        self.target_token_ids = build_target_token_ids(
            self.vocab,
            event_type=cfg.event_type,
            concept_ids=cfg.concept_ids,
            token_regexes=cfg.token_regexes,
        )
        if not self.target_token_ids:
            raise ValueError("No target tokens resolved. Provide --event_type with --concept_ids or --token_regexes.")

        os.makedirs(cfg.output_dir, exist_ok=True)

    # --------------
    # Index time
    # --------------
    def pick_index_time(self, events: List[Dict]) -> Optional[datetime]:
        # events include a 'static' then chronological clinical events; ensure sorted
        clinical = [e for e in events if e.get("event_type") != "static" and isinstance(e.get("timestamp"), datetime)]
        if not clinical:
            return None
        clinical.sort(key=lambda x: x["timestamp"])  # in case not sorted

        horizon = timedelta(days=self.cfg.forward_horizon_days)
        if self.cfg.index_strategy == "end_minus_horizon":
            last_time = clinical[-1]["timestamp"]
            index_time = last_time - horizon
            # ensure index_time is not before first event
            if index_time <= clinical[0]["timestamp"]:
                # Not enough tail; fallback to midpoint minus half-horizon if possible
                mid_idx = len(clinical) // 2
                index_time = clinical[mid_idx]["timestamp"]
            return index_time
        elif self.cfg.index_strategy == "random":
            # pick a random time where there is at least horizon afterwards
            viable_times = [e["timestamp"] for e in clinical if e["timestamp"] + horizon <= clinical[-1]["timestamp"]]
            if not viable_times:
                return clinical[0]["timestamp"]
            rng_idx = np.random.randint(0, len(viable_times))
            return viable_times[rng_idx]
        else:
            # default to start
            return clinical[0]["timestamp"]

    # --------------
    # Tokenization up to index time
    # --------------
    def tokenize_up_to(self, events: List[Dict], index_time: datetime) -> List[int]:
        static_event = next((e for e in events if e.get("event_type") == "static"), None)
        birth_year = None
        if static_event is not None:
            birth_year = static_event.get("birth_year")
        patient_age = 50
        if birth_year is not None and isinstance(birth_year, int):
            try:
                patient_age = max(0, index_time.year - birth_year)
            except Exception:
                patient_age = 50

        sub_events = [e for e in events if e.get("event_type") == "static" or (isinstance(e.get("timestamp"), datetime) and e["timestamp"] <= index_time)]
        sub_events.sort(key=lambda x: x.get("timestamp", datetime.min))

        tokens = self.processor.tokenize_timeline(sub_events, patient_age)
        return tokens

    # --------------
    # Ground truth in horizon
    # --------------
    def event_in_horizon(self, events: List[Dict], index_time: datetime) -> int:
        horizon_end = index_time + timedelta(days=self.cfg.forward_horizon_days)
        # Determine if any target concept appears in [index_time, horizon_end]
        target_prefixes = []
        # Derive textual prefixes for quick checking
        if self.cfg.event_type and self.cfg.concept_ids:
            if self.cfg.event_type.lower() == "condition":
                target_prefixes = [f"CONDITION_{cid}" for cid in self.cfg.concept_ids]
            elif self.cfg.event_type.lower() == "medication":
                target_prefixes = [f"DRUG_{cid}" for cid in self.cfg.concept_ids]
            elif self.cfg.event_type.lower() == "procedure":
                target_prefixes = [f"PROCEDURE_{cid}" for cid in self.cfg.concept_ids]
            elif self.cfg.event_type.lower() == "measurement":
                target_prefixes = [f"MEASUREMENT_{cid}" for cid in self.cfg.concept_ids]

        for e in events:
            if e.get("event_type") == "static":
                continue
            ts: Optional[datetime] = e.get("timestamp")
            if ts is None:
                continue
            if index_time <= ts <= horizon_end:
                # test against prefixes if we have them; else fall back to regexes over vocab unavailable here
                if target_prefixes:
                    # Identify the concept token name this event would produce
                    if e.get("event_type") == "condition":
                        name = f"CONDITION_{e.get('condition_concept_id', 'unknown')}"
                    elif e.get("event_type") == "medication":
                        name = f"DRUG_{e.get('drug_concept_id', 'unknown')}"
                    elif e.get("event_type") == "procedure":
                        name = f"PROCEDURE_{e.get('procedure_concept_id', 'unknown')}"
                    elif e.get("event_type") == "measurement":
                        name = f"MEASUREMENT_{e.get('measurement_concept_id', 'unknown')}"
                    else:
                        name = None
                    if name and any(name == tp for tp in target_prefixes):
                        return 1
                else:
                    # If only regex-based selection was provided, approximate by matching constructed names
                    # Build a set of compiled regexes from cfg.token_regexes
                    if self.cfg.token_regexes:
                        constructed: List[str] = []
                        if e.get("event_type") == "condition":
                            constructed.append(f"CONDITION_{e.get('condition_concept_id', 'unknown')}")
                        if e.get("event_type") == "medication":
                            constructed.append(f"DRUG_{e.get('drug_concept_id', 'unknown')}")
                        if e.get("event_type") == "procedure":
                            constructed.append(f"PROCEDURE_{e.get('procedure_concept_id', 'unknown')}")
                        if e.get("event_type") == "measurement":
                            constructed.append(f"MEASUREMENT_{e.get('measurement_concept_id', 'unknown')}")
                        for name in constructed:
                            if any(re.search(rx, name) for rx in self.cfg.token_regexes):
                                return 1
        return 0

    # --------------
    # Simulation within horizon
    # --------------
    def simulate_probability(self, history_tokens: List[int]) -> float:
        """Estimate probability of target event in horizon via Monte Carlo forward generation."""
        horizon_minutes = self.cfg.forward_horizon_days * 24 * 60
        count = 0
        input_ids = torch.tensor([history_tokens], dtype=torch.long, device=self.device)
        for _ in range(self.cfg.num_samples):
            generated = self._generate(input_ids)
            if self._generated_contains_event_within_horizon(generated):
                count += 1
        return count / float(self.cfg.num_samples)

    def _generate(self, input_ids: torch.Tensor) -> List[int]:
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_length=self.cfg.max_gen_tokens,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                top_p=self.cfg.top_p,
                do_sample=True,
            )
        # return only the continuation tokens (flattened list of ids)
        return out[0, input_ids.size(1):].tolist()

    def _generated_contains_event_within_horizon(self, gen_tokens: List[int]) -> bool:
        minutes = 0
        for tid in gen_tokens:
            name = self.id_to_token.get(tid, "")
            # Check for target event occurrence
            if tid in self.target_token_ids:
                # event occurred before surpassing horizon
                return True
            # Accumulate time when we see time tokens
            delta = token_minutes_from_name(name)
            if delta is not None:
                minutes += delta
                if minutes >= self.cfg.forward_horizon_days * 24 * 60:
                    return False
        # Horizon not reached within max_gen_tokens; treat as no event within horizon
        return False

    # --------------
    # End-to-end evaluation
    # --------------
    def run(self) -> None:
        patient_ids = list(self.patient_timelines.keys())
        if self.cfg.patient_limit is not None:
            patient_ids = patient_ids[: self.cfg.patient_limit]

        y_true: List[int] = []
        y_prob: List[float] = []
        per_patient: Dict[int, Dict] = {}

        for idx, pid in enumerate(patient_ids, start=1):
            events = self.patient_timelines[pid]
            index_time = self.pick_index_time(events)
            if index_time is None:
                continue
            # Build history tokens
            history_tokens = self.tokenize_up_to(events, index_time)
            # Ground truth label from real events within horizon
            label = self.event_in_horizon(events, index_time)
            # Predicted probability from simulation
            prob = self.simulate_probability(history_tokens)

            y_true.append(label)
            y_prob.append(prob)
            per_patient[pid] = {
                "index_time": index_time.isoformat(),
                "label": int(label),
                "prob": float(prob),
                "history_len": len(history_tokens),
            }

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(patient_ids)} patients...")

        if not y_true:
            print("No evaluable patients.")
            return

        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)

        # Metrics
        try:
            auroc = roc_auc_score(y_true_arr, y_prob_arr)
        except Exception:
            auroc = float("nan")
        try:
            auprc = average_precision_score(y_true_arr, y_prob_arr)
        except Exception:
            auprc = float("nan")
        try:
            brier = brier_score_loss(y_true_arr, y_prob_arr)
        except Exception:
            brier = float("nan")
        ece, bin_acc, bin_conf = expected_calibration_error(y_true_arr, y_prob_arr, n_bins=10)

        results = {
            "n_patients": int(len(y_true)),
            "auroc": float(auroc),
            "auprc": float(auprc),
            "brier": float(brier),
            "ece_10": float(ece),
        }

        # Save per-patient and summary
        with open(os.path.join(self.cfg.output_dir, "tte_predictions.json"), "w") as f:
            json.dump(per_patient, f, indent=2)
        with open(os.path.join(self.cfg.output_dir, "tte_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Calibration plot
        self._plot_calibration(bin_acc, bin_conf)

        # Print summary
        print(json.dumps(results, indent=2))

    def _plot_calibration(self, bin_acc: np.ndarray, bin_conf: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        centers = np.linspace(0.05, 0.95, len(bin_acc))
        ax.plot(centers, bin_acc, marker="o", label="Empirical")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title("Calibration (Reliability) Curve")
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, "calibration.png"), dpi=200)
        plt.close(fig)


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate time-to-event via forward simulation")
    p.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (best or latest)")
    p.add_argument("--data_dir", type=str, default=data_config.output_dir, help="Processed data directory")
    p.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation outputs")
    p.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    p.add_argument("--forward_horizon_days", type=int, default=365)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--max_gen_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--index_strategy", type=str, choices=["end_minus_horizon", "random"], default="end_minus_horizon")
    p.add_argument("--event_type", type=str, choices=["condition", "medication", "procedure", "measurement"], default=None)
    p.add_argument("--concept_ids", type=str, default=None, help="Comma-separated concept IDs (e.g., OMOP) matching event_type")
    p.add_argument("--token_regex", action="append", default=None, help="Regex over vocab token names for matching target tokens; can be specified multiple times")
    p.add_argument("--patient_limit", type=int, default=None)
    args = p.parse_args()

    concept_ids_list: Optional[List[int]] = None
    if args.concept_ids:
        concept_ids_list = [int(x) for x in re.split(r"[,\s]+", args.concept_ids.strip()) if x]

    return EvalConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        forward_horizon_days=args.forward_horizon_days,
        num_samples=args.num_samples,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        index_strategy=args.index_strategy,
        event_type=args.event_type,
        concept_ids=concept_ids_list,
        token_regexes=args.token_regex,
        patient_limit=args.patient_limit,
    )


def main() -> None:
    cfg = parse_args()
    evaluator = TimeToEventEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()


