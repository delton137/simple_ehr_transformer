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
import torch.nn.functional as F
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
        if len(self.current_timelines) == 0:
            # Diagnostics: check for tokenized timelines to infer available patient ids
            tt_path = os.path.join(current_data_dir, "tokenized_timelines.pkl")
            if os.path.exists(tt_path):
                try:
                    with open(tt_path, "rb") as f:
                        import pickle
                        _tt = pickle.load(f)
                    _ids = list(_tt.keys())
                    try:
                        _ids = [int(k) for k in _ids]
                    except Exception:
                        pass
                    print(f"Info: current_data_dir tokenized_timelines.pkl has {len(_ids)} patient IDs (patient_timelines.pkl is empty)")
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
                if len(self.future_timelines) == 0:
                    tt_path2 = os.path.join(future_data_dir, "tokenized_timelines.pkl")
                    if os.path.exists(tt_path2):
                        try:
                            with open(tt_path2, "rb") as f:
                                import pickle
                                _tt2 = pickle.load(f)
                            _ids2 = list(_tt2.keys())
                            try:
                                _ids2 = [int(k) for k in _ids2]
                            except Exception:
                                pass
                            print(f"Info: future_data_dir tokenized_timelines.pkl has {len(_ids2)} patient IDs (patient_timelines.pkl is empty)")
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

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


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
    p.add_argument("--output_dir", type=str, default=None, help="Output directory; if not set, uses the directory of --model_path")
    p.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    p.add_argument("--forward_horizon_days", type=int, default=365)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--max_gen_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--patient_limit", type=int, default=None)
    p.add_argument("--targets", type=str, default=None, help="Comma-separated list of target token names, e.g. CONDITION_201826,CONDITION_444070,CONDITION_316139")
    p.add_argument("--vocab_path", type=str, default=None, help="Optional explicit path to vocabulary.pkl matching the checkpoint")
    return p.parse_args()


def main() -> None:
    import argparse
    import pickle
    import time
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')

    p = argparse.ArgumentParser(description="Evaluate next-year concept occurrence using tokenized timelines")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--current_data_dir", type=str, required=True, help="Processed data dir for current year (e.g., pre_2023)")
    p.add_argument("--future_data_dir", type=str, required=True, help="Processed data dir for next year (e.g., 2023)")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--max_input_len", type=int, default=512)
    p.add_argument("--max_gen_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--patient_limit", type=int, default=None)
    p.add_argument("--targets", type=str, required=True)
    p.add_argument("--debug_samples", type=int, default=0, help="Print detailed debug for first N patients")
    p.add_argument("--debug_tokens", type=int, default=50, help="How many tokens to print for history/generated samples in debug")
    args = p.parse_args()

    # Resolve default output_dir if not provided
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.model_path) or "."

    # Load current vocab and tokenized timelines
    with open(os.path.join(args.current_data_dir, "vocabulary.pkl"), "rb") as f:
        vocab: Dict[str, int] = pickle.load(f)
    with open(os.path.join(args.current_data_dir, "tokenized_timelines.pkl"), "rb") as f:
        current_tt: Dict[int, List[int]] = pickle.load(f)

    # Load future tokenized timelines
    with open(os.path.join(args.future_data_dir, "tokenized_timelines.pkl"), "rb") as f:
        future_tt: Dict[int, List[int]] = pickle.load(f)
    # Load future vocabulary for correct decoding of future IDs
    with open(os.path.join(args.future_data_dir, "vocabulary.pkl"), "rb") as f:
        future_vocab: Dict[str, int] = pickle.load(f)
    id_to_token_future = {tid: name for name, tid in future_vocab.items()}

    # Overlap of patient IDs
    cur_ids = set(current_tt.keys())
    fut_ids = set(future_tt.keys())
    overlap = sorted(cur_ids & fut_ids)
    print(f"Overlapping patients: {len(overlap)} (Current={len(cur_ids)}, Future={len(fut_ids)})")

    # Model
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    model = create_ethos_model(len(vocab))
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[index]
    model = model.to(device)
    model.eval()

    id_to_token = {tid: name for name, tid in vocab.items()}

    # Targets: allow user-provided list; add legacy/table-inclusive variants automatically
    targets: List[str] = [t.strip() for t in args.targets.split(',') if t.strip()]

    target_variants: Dict[str, set] = {}
    def expand_variants(token_name: str) -> set:
        s = {token_name}
        # CONDITION
        if token_name.startswith("CONDITION_OCCURRENCE_"):
            s.add(token_name.replace("CONDITION_OCCURRENCE_", "CONDITION_", 1))
        if token_name.startswith("CONDITION_"):
            s.add(token_name.replace("CONDITION_", "CONDITION_OCCURRENCE_", 1))
        # DRUG
        if token_name.startswith("DRUG_EXPOSURE_"):
            s.add(token_name.replace("DRUG_EXPOSURE_", "DRUG_", 1))
        if token_name.startswith("DRUG_"):
            s.add(token_name.replace("DRUG_", "DRUG_EXPOSURE_", 1))
        # PROCEDURE
        if token_name.startswith("PROCEDURE_OCCURRENCE_"):
            s.add(token_name.replace("PROCEDURE_OCCURRENCE_", "PROCEDURE_", 1))
        if token_name.startswith("PROCEDURE_"):
            s.add(token_name.replace("PROCEDURE_", "PROCEDURE_OCCURRENCE_", 1))
        return s
    for t in targets:
        target_variants[t] = expand_variants(t)

    def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        """Apply temperature, top-k and top-p (nucleus) sampling and return next token ids (batch x 1)."""
        next_token_logits = logits[:, -1, :] / max(1e-6, temperature)
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            filtered = torch.full_like(next_token_logits, -float('inf'))
            filtered.scatter_(1, top_k_indices, top_k_logits)
            next_token_logits = filtered
        probs = F.softmax(next_token_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _generate_until_horizon(history_ids: List[int]) -> List[int]:
        """Autoregressively generate up to max_gen_tokens or 1-year via TIME_* tokens."""
        if not history_ids:
            return []
        # Seed with tail of history
        current_ids = torch.tensor([history_ids[-args.max_input_len:]], dtype=torch.long, device=device)
        generated: List[int] = []
        minutes = 0
        horizon = 365 * 24 * 60
        with torch.no_grad():
            for _ in range(args.max_gen_tokens):
                logits = model(current_ids)
                next_token = _sample_next_token(logits, args.temperature, args.top_k)
                tid = int(next_token.item())
                generated.append(tid)
                # Update horizon
                name = id_to_token.get(tid, "")
                if name.startswith("TIME_"):
                    label = name.split("TIME_")[-1]
                    delta = TIME_LABEL_TO_MINUTES.get(label)
                    if delta is not None:
                        minutes += delta
                    if minutes >= horizon:
                        break
                # Append to context
                current_ids = torch.cat([current_ids, next_token], dim=1)
                # Early end if EOS (look up by name to avoid self)
                if name == '<EOS>':
                    break
        return generated

    def generate_probs(history_ids: List[int]) -> Dict[str, float]:
        probs: Dict[str, float] = {t: 0.0 for t in targets}
        if not history_ids:
            return probs
        count_present: Dict[str, int] = {t: 0 for t in targets}
        for _ in range(args.num_samples):
            cont = _generate_until_horizon(history_ids)
            names_in_horizon: list[str] = [id_to_token.get(tid, "") for tid in cont]
            name_set = set(names_in_horizon)
            for key, vars_set in target_variants.items():
                if any(v in name_set for v in vars_set):
                    count_present[key] += 1
        for key in targets:
            probs[key] = count_present[key] / float(args.num_samples)
        return probs

    # Prepare labels from future tokenized timelines (presence in the year)
    STRUCT_PREFIXES = ("AGE_", "GENDER_", "RACE_", "YEAR_")
    def label_from_future(seq: List[int]) -> Dict[str, int]:
        # Time-bounded presence within 1 year, skipping leading structural tokens
        horizon = 365 * 24 * 60
        minutes = 0
        started = False
        lab: Dict[str, int] = {k: 0 for k in target_variants.keys()}
        remaining = set(target_variants.keys())
        for tid in seq:
            name = id_to_token_future.get(tid, "")
            if not started:
                if not name or name.startswith(STRUCT_PREFIXES):
                    continue
                started = True
            if name.startswith("TIME_"):
                label = name.split("TIME_")[-1]
                delta = TIME_LABEL_TO_MINUTES.get(label)
                if delta is not None:
                    minutes += delta
                if minutes > horizon:
                    break
                continue
            # Check targets
            if remaining:
                for key in list(remaining):
                    if name in target_variants[key]:
                        lab[key] = 1
                        remaining.discard(key)
                        break
        return lab

    y_prob: Dict[str, List[float]] = {t: [] for t in targets}
    y_true: Dict[str, List[int]] = {t: [] for t in targets}

    # Balanced sampling across multiple conditions: for each target, sample ~N//Y//2 positives and ~N//Y//2 negatives
    import random
    def has_future_token(pid: int, token_key: str) -> bool:
        # Use time-bounded labeling logic for balancing as well
        labs = label_from_future(future_tt.get(pid, []))
        return labs.get(token_key, 0) == 1

    if args.patient_limit is not None and len(overlap) > args.patient_limit:
        want = args.patient_limit
        per_cond_half = max(1, want // max(1, len(targets)) // 2)
        selected_set = set()
        for tok in targets:
            pos_pids = [pid for pid in overlap if has_future_token(pid, tok)]
            neg_pids = [pid for pid in overlap if not has_future_token(pid, tok)]
            pos_take = min(per_cond_half, len(pos_pids))
            neg_take = min(per_cond_half, len(neg_pids))
            selected_set.update(random.sample(pos_pids, pos_take))
            selected_set.update(random.sample(neg_pids, neg_take))
        # Top up to desired size if needed
        if len(selected_set) < want:
            remaining = [pid for pid in overlap if pid not in selected_set]
            extra_take = min(want - len(selected_set), len(remaining))
            selected_set.update(random.sample(remaining, extra_take))
        selected = list(selected_set)
        random.shuffle(selected)
    else:
        selected = overlap

    per_patient: Dict[int, Dict[str, float]] = {}
    total = len(selected)
    # Report target prevalence in the selected test set (time-bounded labels)
    counts_in_test: Dict[str, int] = {t: 0 for t in targets}
    for pid in selected:
        labs = label_from_future(future_tt.get(pid, []))
        for t in targets:
            counts_in_test[t] += int(labs.get(t, 0))
    print("Target prevalence in test set:")
    for t in targets:
        print(f"  {t}: {counts_in_test[t]} / {total}")
    print(f"Evaluating on {total} patients (balanced across {len(targets)} conditions where possible)")
    for i, pid in enumerate(selected, start=1):
        t0 = time.time()
        hist = current_tt.get(pid, [])
        fut = future_tt.get(pid, [])
        probs = generate_probs(hist)
        labs = label_from_future(fut)
        per_patient[pid] = probs
        for key in targets:
            y_prob[key].append(probs.get(key, 0.0))
            y_true[key].append(labs.get(key, 0))
        # Optional debug printing
        if args.debug_samples and i <= args.debug_samples:
            def decode_seq(ids: List[int], mapper: Dict[int, str]) -> List[str]:
                return [mapper.get(tid, "") for tid in ids]
            hist_names = decode_seq(hist[-args.debug_tokens:], id_to_token)
            # Single-sample generation preview
            gen_names: List[str] = []
            try:
                cont_dbg = _generate_until_horizon(hist)
                gen_names = [id_to_token.get(tid, "") for tid in cont_dbg[:args.debug_tokens]]
            except Exception:
                pass
            # Future preview without leading structural tokens
            fut_names_raw = decode_seq(fut, id_to_token_future)
            j = 0
            while j < len(fut_names_raw) and fut_names_raw[j].startswith(STRUCT_PREFIXES):
                j += 1
            fut_names = fut_names_raw[j:j + args.debug_tokens]
            print(f"=== DEBUG patient {pid} ===")
            print(f"history_tail ({len(hist)} total): {hist_names}")
            print(f"generated_preview: {gen_names}")
            print(f"future_preview: {fut_names}")
            print(f"labels: {labs}")
        dt = time.time() - t0
        print(f"completed patient {i}/{total} time={dt:.2f} seconds")

    # AUROC per target
    aurocs: Dict[str, float] = {}
    for key in targets:
        try:
            aurocs[key] = roc_auc_score(y_true[key], y_prob[key])
        except Exception:
            aurocs[key] = float("nan")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "per_patient_probs.json"), "w") as f:
        json.dump(per_patient, f, indent=2)
    with open(os.path.join(args.output_dir, "auroc_per_token.json"), "w") as f:
        json.dump(aurocs, f, indent=2)

    # Plot ROC curves per target
    def _safe_name(s: str) -> str:
        return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in s)
    for key in targets:
        y = y_true[key]
        p = y_prob[key]
        if not y or (sum(y) == 0 or sum(y) == len(y)):
            continue
        fpr, tpr, _ = roc_curve(y, p)
        auc_val = auc(fpr, tpr)
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}", linewidth=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC: {key}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"roc_{_safe_name(key)}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

    print("AUROC per token:")
    for k, v in aurocs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


