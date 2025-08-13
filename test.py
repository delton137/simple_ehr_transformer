#!/usr/bin/env python3
"""
Inference and testing harness:

- Runs forward generation on current patients' histories to produce multiple
  simulated future timelines within a given horizon (e.g., 1 year).
- Aggregates predicted future clinical events and their probabilities
  (fraction of simulations in which they occur within horizon).
- If a future OMOP dataset is provided, aligns by patient and compares predicted
  events against ground-truth events within the horizon window, reporting
  micro-averaged precision/recall/F1 and Jaccard.

Outputs per-patient JSON artifacts with generated futures and aggregated predictions,
and a summary metrics JSON if ground truth is available.
"""

import os
import json
import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from model import create_ethos_model
from sklearn.metrics import roc_auc_score


# -----------------------------
# Time utilities
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


def event_token_name_from_event(e: Dict) -> Optional[str]:
    et = e.get("event_type")
    if et == "condition":
        # Prefer table-inclusive prefix; retain legacy as fallback handled elsewhere
        return f"CONDITION_OCCURRENCE_{e.get('condition_concept_id', 'unknown')}"
    if et == "medication":
        return f"DRUG_EXPOSURE_{e.get('drug_concept_id', 'unknown')}"
    if et == "procedure":
        return f"PROCEDURE_OCCURRENCE_{e.get('procedure_concept_id', 'unknown')}"
    if et == "measurement":
        return f"MEASUREMENT_{e.get('measurement_concept_id', 'unknown')}"
    if et == "observation":
        return f"OBSERVATION_{e.get('observation_concept_id', 'unknown')}"
    return None


# -----------------------------
# Core tester
# -----------------------------

class FutureTester:
    def __init__(
        self,
        model_path: str,
        current_data_dir: str,
        output_dir: str,
        device: str = "auto",
        future_data_dir: Optional[str] = None,
        forward_horizon_days: int = 365,
        num_samples: int = 50,
        max_gen_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        patient_limit: Optional[int] = None,
        vocab_path: Optional[str] = None,
    ) -> None:
        self.forward_horizon_days = forward_horizon_days
        self.num_samples = num_samples
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.patient_limit = patient_limit

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint first to know its expected vocab size
        checkpoint = torch.load(model_path, map_location=self.device)
        ckpt_state = checkpoint.get("model_state_dict", {})
        ckpt_vocab_size: Optional[int] = None
        try:
            ckpt_vocab_size = int(ckpt_state["token_embedding.weight"].size(0))  # type: ignore[index]
        except Exception:
            ckpt_vocab_size = None

        # Resolve vocabulary path preference:
        # 1) explicit --vocab_path
        # 2) vocabulary.pkl next to checkpoint
        # 3) vocabulary.pkl in current_data_dir (fallback)
        vocab_candidates = []
        if vocab_path:
            vocab_candidates.append(vocab_path)
        model_dir = os.path.dirname(model_path)
        vocab_candidates.append(os.path.join(model_dir, "vocabulary.pkl"))
        vocab_candidates.append(os.path.join(current_data_dir, "vocabulary.pkl"))

        chosen_vocab_path: Optional[str] = None
        for vp in vocab_candidates:
            if vp and os.path.exists(vp):
                chosen_vocab_path = vp
                break

        if chosen_vocab_path is None:
            raise FileNotFoundError(
                "Could not find a vocabulary.pkl. Provide --vocab_path, or place vocabulary.pkl next to the checkpoint, or ensure it exists in current_data_dir."
            )

        with open(chosen_vocab_path, "rb") as f:
            import pickle
            self.vocab: Dict[str, int] = pickle.load(f)
        self.id_to_token = {tid: name for name, tid in self.vocab.items()}

        # Load current patient timelines (events with timestamps)
        with open(os.path.join(current_data_dir, "patient_timelines.pkl"), "rb") as f:
            import pickle
            self.current_timelines: Dict[int, List[Dict]] = pickle.load(f)
        # Normalize patient ID key types to int
        try:
            self.current_timelines = {int(k): v for k, v in self.current_timelines.items()}
        except Exception:
            pass

        # Optional: future ground-truth timelines
        self.future_timelines: Optional[Dict[int, List[Dict]]] = None
        if future_data_dir is not None:
            fut_path = os.path.join(future_data_dir, "patient_timelines.pkl")
            if os.path.exists(fut_path):
                with open(fut_path, "rb") as f:
                    import pickle
                    self.future_timelines = pickle.load(f)
                # Normalize patient ID key types to int
                try:
                    self.future_timelines = {int(k): v for k, v in self.future_timelines.items()}  # type: ignore[dict-item]
                except Exception:
                    pass
        
        # Quick diagnostics if a future dataset exists
        if self.future_timelines is not None:
            cur_ids = set(self.current_timelines.keys())
            fut_ids = set(self.future_timelines.keys())
            overlap = len(cur_ids & fut_ids)
            if overlap == 0:
                print(
                    f"Warning: 0 overlapping patients. Current={len(cur_ids)}, Future={len(fut_ids)}. "
                    f"ID type examples: current={type(next(iter(cur_ids))).__name__ if cur_ids else 'NA'}, "
                    f"future={type(next(iter(fut_ids))).__name__ if fut_ids else 'NA'}"
                )

        # Load model and adapt checkpoint to current vocab size if needed
        self.model = create_ethos_model(len(self.vocab))
        state_dict = checkpoint.get("model_state_dict", {})
        if ckpt_vocab_size is not None and ckpt_vocab_size != len(self.vocab):
            # Adapt embedding and output layers by copying overlapping rows/cols
            model_state = self.model.state_dict()
            def _resize_param(key: str) -> None:
                if key not in state_dict or key not in model_state:
                    return
                src = state_dict[key]
                dst = model_state[key]
                if src.shape == dst.shape:
                    return
                new_param = dst.clone()
                # Handle 2D and 1D tensors
                if src.dim() == 2 and dst.dim() == 2:
                    rows = min(src.size(0), dst.size(0))
                    cols = min(src.size(1), dst.size(1))
                    new_param[:rows, :cols] = src[:rows, :cols]
                elif src.dim() == 1 and dst.dim() == 1:
                    n = min(src.size(0), dst.size(0))
                    new_param[:n] = src[:n]
                else:
                    # Fallback: if dims differ, skip
                    return
                state_dict[key] = new_param

            _resize_param("token_embedding.weight")
            _resize_param("output_projection.weight")
            _resize_param("output_projection.bias")
        self.model.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]
        self.model = self.model.to(self.device)
        self.model.eval()

        # Output dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Hard-coded tokens to predict (Type 2 diabetes). Include legacy and table-inclusive.
        # User provided: CONDITION_201826 (correct); earlier typo 20182 ignored.
        self.tokens_to_predict: List[str] = [
            "CONDITION_OCCURRENCE_201826",
            "CONDITION_201826",
        ]
        # Map to actual vocab names present; allow flexible matching
        self.token_variants: Dict[str, Set[str]] = self._resolve_token_variants(self.tokens_to_predict)

    def _resolve_token_variants(self, desired_tokens: List[str]) -> Dict[str, Set[str]]:
        """For each desired token, include both legacy and table-inclusive variants.
        Does not require that variants exist in vocab; matching is done on names.
        """
        variants: Dict[str, Set[str]] = {}
        pairs = [
            ("CONDITION_OCCURRENCE_", "CONDITION_"),
            ("DRUG_EXPOSURE_", "DRUG_"),
            ("PROCEDURE_OCCURRENCE_", "PROCEDURE_"),
        ]
        for tok in desired_tokens:
            s: Set[str] = set([tok])
            for inc, leg in pairs:
                if tok.startswith(inc):
                    s.add(tok.replace(inc, leg, 1))
                if tok.startswith(leg):
                    s.add(tok.replace(leg, inc, 1))
            variants[tok] = s
        return variants

    def _truth_occurrence_for_targets(self, current_events: List[Dict], future_events: List[Dict]) -> Dict[str, int]:
        """Return a map canonical_token -> 0/1 indicating any occurrence in the horizon."""
        index_time = self.last_event_time(current_events)
        if index_time is None:
            return {k: 0 for k in self.token_variants.keys()}
        horizon_end = index_time + timedelta(days=self.forward_horizon_days)
        names_in_horizon: Set[str] = set()
        for e in future_events:
            if e.get("event_type") == "static":
                continue
            ts = e.get("timestamp")
            if not isinstance(ts, datetime):
                continue
            if index_time < ts <= horizon_end:
                nm = event_token_name_from_event(e)
                if nm:
                    names_in_horizon.add(nm)
        out: Dict[str, int] = {}
        for canonical, alts in self.token_variants.items():
            out[canonical] = 1 if any(n in names_in_horizon for n in alts) else 0
        return out

    # -----------------
    # Index and history
    # -----------------
    def last_event_time(self, events: List[Dict]) -> Optional[datetime]:
        times = [e.get("timestamp") for e in events if e.get("event_type") != "static" and isinstance(e.get("timestamp"), datetime)]
        if not times:
            return None
        return max(times)

    def build_history_tokens(self, events: List[Dict], index_time: datetime) -> List[int]:
        # Reuse tokenization logic indirectly by reproducing time and event tokens from events up to index_time
        # Here we construct tokens using the same token names the vocab expects.
        # For simplicity in this harness, we build a minimal history by taking the full tokenized timeline saved earlier if available.
        # However, we only have raw events here; so we will approximate by using ids of tokens present in vocab as event tokens without re-creating time deltas.
        # A more faithful approach is available in data_processor. For testing forward generation, providing recent tail should suffice.
        # We'll include static tokens if present.
        tokens: List[int] = []

        # Static tokens (approximate: include AGE if available else skip)
        # We skip static to avoid mismatch; generation works without them.

        # Chronological events up to index_time
        chronological = [e for e in events if e.get("event_type") != "static" and isinstance(e.get("timestamp"), datetime) and e["timestamp"] <= index_time]
        chronological.sort(key=lambda x: x["timestamp"])
        # Only take the last N events for context to fit within max length
        tail = chronological[-256:]
        for e in tail:
            name = event_token_name_from_event(e)
            if name is None:
                continue
            tid = self.vocab.get(name)
            if tid is not None:
                tokens.append(tid)
        # Append EOS if known (optional)
        eos_id = self.vocab.get("<EOS>")
        if eos_id is not None:
            tokens.append(eos_id)
        return tokens

    # -----------------
    # Generation
    # -----------------
    def generate_future_once(self, history_ids: List[int]) -> List[Tuple[str, int]]:
        """Generate one simulated future. Returns list of (token_name, minutes_since_start) within generation window."""
        input_ids = torch.tensor([history_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_length=self.max_gen_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=True,
            )
        cont = out[0, input_ids.size(1):].tolist()
        minutes = 0
        events_with_time: List[Tuple[str, int]] = []
        horizon_minutes = self.forward_horizon_days * 24 * 60
        for tid in cont:
            name = self.id_to_token.get(tid, f"UNK_{tid}")
            # accumulate time
            delta = token_minutes_from_name(name)
            if delta is not None:
                minutes += delta
                if minutes >= horizon_minutes:
                    break
                continue
            # record clinical events
            if name.startswith((
                "CONDITION_", "CONDITION_OCCURRENCE_",
                "DRUG_", "DRUG_EXPOSURE_",
                "PROCEDURE_", "PROCEDURE_OCCURRENCE_",
                "MEASUREMENT_", "OBSERVATION_",
            )):
                events_with_time.append((name, minutes))
        return events_with_time

    def simulate_patient(self, pid: int, events: List[Dict]) -> Dict:
        index_time = self.last_event_time(events)
        if index_time is None:
            return {"pid": pid, "skipped": True}
        history_ids = self.build_history_tokens(events, index_time)
        sims: List[List[Tuple[str, int]]] = []
        for _ in range(self.num_samples):
            sims.append(self.generate_future_once(history_ids))

        # Aggregate probabilities per token name
        token_occurrence_counts: Dict[str, int] = {}
        for sim in sims:
            seen = {name for (name, _t) in sim}
            for name in seen:
                token_occurrence_counts[name] = token_occurrence_counts.get(name, 0) + 1
        token_prob = {name: cnt / float(self.num_samples) for name, cnt in token_occurrence_counts.items()}

        # Compute probabilities for tokens_to_predict (per canonical key)
        targeted_prob: Dict[str, float] = {}
        for canonical, variants in self.token_variants.items():
            count = 0
            for sim in sims:
                present = any(name in variants for (name, _t) in sim)
                if present:
                    count += 1
            targeted_prob[canonical] = count / float(self.num_samples)

        return {
            "pid": pid,
            "index_time": index_time.isoformat(),
            "history_len": len(history_ids),
            "simulations": sims,
            "predicted_event_prob": token_prob,
            "targeted_prob": targeted_prob,
        }

    # -----------------
    # Ground-truth comparison
    # -----------------
    def extract_future_truth(self, pid: int, current_events: List[Dict], future_events: List[Dict]) -> List[str]:
        """Return list of ground-truth event token names occurring within horizon after the current index time."""
        index_time = self.last_event_time(current_events)
        if index_time is None:
            return []
        horizon_end = index_time + timedelta(days=self.forward_horizon_days)
        truth: List[str] = []
        for e in future_events:
            if e.get("event_type") == "static":
                continue
            ts = e.get("timestamp")
            if not isinstance(ts, datetime):
                continue
            if index_time < ts <= horizon_end:
                name = event_token_name_from_event(e)
                if name:
                    truth.append(name)
        return truth

    def compute_set_metrics(self, pred_names: List[str], truth_names: List[str]) -> Tuple[float, float, float, float]:
        pred_set = set(pred_names)
        truth_set = set(truth_names)
        tp = len(pred_set & truth_set)
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        jaccard = len(pred_set & truth_set) / len(pred_set | truth_set) if (pred_set | truth_set) else 0.0
        return precision, recall, f1, jaccard

    # -----------------
    # Run
    # -----------------
    def run(self) -> None:
        pids = list(self.current_timelines.keys())
        if self.future_timelines is not None:
            pids = [pid for pid in pids if pid in self.future_timelines]
            print(f"Evaluating on {len(pids)} patients present in both current and future datasets")
        if self.patient_limit is not None:
            pids = pids[: self.patient_limit]

        per_patient_outputs: Dict[int, Dict] = {}
        metrics_accum = {"tp": 0, "fp": 0, "fn": 0, "jacc": []}

        for i, pid in enumerate(pids, start=1):
            cur_events = self.current_timelines[pid]
            out = self.simulate_patient(pid, cur_events)
            per_patient_outputs[pid] = out

            if self.future_timelines is not None and pid in self.future_timelines:
                fut_events = self.future_timelines[pid]
                truth_names = self.extract_future_truth(pid, cur_events, fut_events)
                # Select predicted names above threshold (any occurrence)
                pred_prob = out["predicted_event_prob"]
                pred_names = [name for name, prob in pred_prob.items() if prob > 0.0]
                # Metrics
                precision, recall, f1, jacc = self.compute_set_metrics(pred_names, truth_names)
                # Accumulate micro counts
                pred_set = set(pred_names)
                truth_set = set(truth_names)
                metrics_accum["tp"] += len(pred_set & truth_set)
                metrics_accum["fp"] += len(pred_set - truth_set)
                metrics_accum["fn"] += len(truth_set - pred_set)
                metrics_accum["jacc"].append(jacc)
                per_patient_outputs[pid]["truth_event_names"] = truth_names
                per_patient_outputs[pid]["precision"] = precision
                per_patient_outputs[pid]["recall"] = recall
                per_patient_outputs[pid]["f1"] = f1
                per_patient_outputs[pid]["jaccard"] = jacc

            if i % 25 == 0:
                print(f"Processed {i}/{len(pids)} patients...")

        # Save per-patient outputs
        with open(os.path.join(self.output_dir, "future_simulations.json"), "w") as f:
            json.dump(per_patient_outputs, f, indent=2)

        # Save summary metrics if truth available
        if self.future_timelines is not None:
            tp = metrics_accum["tp"]
            fp = metrics_accum["fp"]
            fn = metrics_accum["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            jacc_mean = float(np.mean(metrics_accum["jacc"])) if metrics_accum["jacc"] else 0.0
            summary = {
                "n_patients": len(per_patient_outputs),
                "precision_micro": precision,
                "recall_micro": recall,
                "f1_micro": f1,
                "jaccard_mean": jacc_mean,
            }
            with open(os.path.join(self.output_dir, "summary_metrics.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print(json.dumps(summary, indent=2))
        else:
            print("No future dataset provided; only generated futures were saved.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate future timelines and compare to future OMOP ground truth")
    p.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    p.add_argument("--current_data_dir", type=str, default="processed_data", help="Processed data dir for current patients (e.g., pre_2023)")
    p.add_argument("--future_data_dir", type=str, default=None, help="Processed data dir for future OMOP (e.g., 2023) for ground truth")
    p.add_argument("--output_dir", type=str, default="test_results", help="Output directory for simulations and metrics")
    p.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    p.add_argument("--forward_horizon_days", type=int, default=365)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--max_gen_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--patient_limit", type=int, default=None)
    p.add_argument("--vocab_path", type=str, default=None, help="Optional explicit path to vocabulary.pkl matching the checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tester = FutureTester(
        model_path=args.model_path,
        current_data_dir=args.current_data_dir,
        future_data_dir=args.future_data_dir,
        output_dir=args.output_dir,
        device=args.device,
        forward_horizon_days=args.forward_horizon_days,
        num_samples=args.num_samples,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        patient_limit=args.patient_limit,
        vocab_path=args.vocab_path,
    )
    # Run simulation and compute AUROC for tokens_to_predict using future data
    pids = list(tester.current_timelines.keys())
    if tester.future_timelines is not None:
        pids = [pid for pid in pids if pid in tester.future_timelines]
        print(f"Evaluating on {len(pids)} patients present in both current and future datasets")
    if tester.patient_limit is not None:
        pids = pids[: tester.patient_limit]

    # Storage for predictions and labels per target
    y_prob: Dict[str, List[float]] = {t: [] for t in tester.token_variants.keys()}
    y_true: Dict[str, List[int]] = {t: [] for t in tester.token_variants.keys()}

    per_patient_outputs: Dict[int, Dict] = {}

    for i, pid in enumerate(pids, start=1):
        cur_events = tester.current_timelines[pid]
        out = tester.simulate_patient(pid, cur_events)
        per_patient_outputs[pid] = out

        # Ground-truth from future dataset within horizon
        truth_map: Dict[str, int] = {}
        if tester.future_timelines is not None and pid in tester.future_timelines:
            truth_map = tester._truth_occurrence_for_targets(
                current_events=cur_events,
                future_events=tester.future_timelines[pid],
            )
        else:
            truth_map = {k: 0 for k in tester.token_variants.keys()}

        # Append per-target probabilities and labels
        for tok in tester.token_variants.keys():
            y_prob[tok].append(out["targeted_prob"].get(tok, 0.0))
            y_true[tok].append(int(truth_map.get(tok, 0)))

        if i % 25 == 0:
            print(f"Processed {i}/{len(pids)} patients...")

    # Compute AUROC per token
    metrics: Dict[str, float] = {}
    for tok in tester.token_variants.keys():
        try:
            metrics[tok] = roc_auc_score(y_true[tok], y_prob[tok])
        except Exception:
            metrics[tok] = float("nan")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "per_patient_predictions.json"), "w") as f:
        json.dump(per_patient_outputs, f, indent=2)
    with open(os.path.join(args.output_dir, "auroc_per_token.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("AUROC per token:")
    for tok, auc in metrics.items():
        print(f"  {tok}: {auc}")


if __name__ == "__main__":
    main()


