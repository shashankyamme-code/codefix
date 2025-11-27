"""
AI Attention Debugger - BONUS Challenge (+10%)

Fully implemented & debugged version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import ast
import inspect
import math


# ---------------------------------------------------------
# Helper: parse class source & AST
# ---------------------------------------------------------
def get_source_ast(model_class):
    src = inspect.getsource(model_class)
    return src, ast.parse(src)


# ---------------------------------------------------------
#   ATTENTION BUG DETECTOR
# ---------------------------------------------------------
class AttentionBugDetector:
    def __init__(self):
        self.detected_bugs = []

    # ======================================================
    # MASTER ANALYZER
    # ======================================================
    def analyze_code(self, module) -> List[Dict[str, any]]:
        bugs = []
        if not hasattr(module, "KVCachedMultiHeadAttention"):
            return bugs

        model_class = module.KVCachedMultiHeadAttention

        checkers = [
            self.check_scaling_factor,
            self.check_softmax_dimension,
            self.check_cache_concatenation,
            self.check_dropout_during_inference
        ]

        for fn in checkers:
            bug = fn(model_class)
            if bug:
                bugs.append(bug)

        bugs.extend(self.check_tensor_dimensions(model_class))
        return bugs

    # ======================================================
    # SCALING FACTOR CHECK
    # ======================================================
    def check_scaling_factor(self, model_class):
        src, tree = get_source_ast(model_class)

        if "/ head_dim" in src or "/ d_k" in src:
            return {
                'type': 'scaling_factor',
                'severity': 'critical',
                'location': 'attention score computation',
                'description': 'Incorrect scaling factor detected',
                'suggestion': 'Use scores /= math.sqrt(head_dim)'
            }
        return None

    # ======================================================
    # SOFTMAX DIM CHECK
    # ======================================================
    def check_softmax_dimension(self, model_class):
        src, tree = get_source_ast(model_class)

        if "softmax" in src:
            if "dim=-2" in src:
                return {
                    'type': 'softmax_dim',
                    'severity': 'high',
                    'location': 'softmax(scores)',
                    'description': 'Softmax applied along wrong dimension',
                    'suggestion': 'Use F.softmax(scores, dim=-1)'
                }
        return None

    # ======================================================
    # CACHE CONCAT DIM CHECK
    # ======================================================
    def check_cache_concatenation(self, model_class):
        src, _ = get_source_ast(model_class)

        if "torch.cat" in src:
            if "dim=2" in src:
                return {
                    'type': 'cache_concat',
                    'severity': 'high',
                    'location': 'cache update',
                    'description': 'Cache concatenated along wrong dimension',
                    'suggestion': 'Use torch.cat([...], dim=1)'
                }
        return None

    # ======================================================
    # DROPOUT DURING INFERENCE
    # ======================================================
    def check_dropout_during_inference(self, model_class):
        src, _ = get_source_ast(model_class)

        if "dropout" in src and "self.training" not in src:
            return {
                'type': 'dropout_inference',
                'severity': 'medium',
                'location': 'attention forward()',
                'description': 'Dropout applied during inference',
                'suggestion': 'Wrap dropout with: if self.training:'
            }
        return None

    # ======================================================
    # DIMENSION BUGS
    # ======================================================
    def check_tensor_dimensions(self, model_class):
        src, _ = get_source_ast(model_class)
        bugs = []

        if ".transpose(2, 3)" not in src and "matmul" in src:
            bugs.append({
                'type': 'missing_transpose',
                'severity': 'medium',
                'location': 'QK^T matmul',
                'description': 'Key/value tensors not transposed',
                'suggestion': 'Ensure K = K.transpose(2, 3)'
            })

        if "reshape" in src and "num_heads" in src and "d_head" not in src:
            bugs.append({
                'type': 'reshape_order',
                'severity': 'low',
                'location': 'projection reshape',
                'description': 'Potentially incorrect reshape ordering',
                'suggestion': 'Use (batch, seq, num_heads, head_dim)'
            })

        return bugs

    # ======================================================
    # FIX SUGGESTION
    # ======================================================
    def suggest_fix(self, bug):
        return f"Suggested fix: {bug['suggestion']}"

    # ======================================================
    # RUNNER
    # ======================================================
    def run_analysis(self, module) -> None:
        print("=" * 70)
        print("AI Attention Debugger - Analysis Report")
        print("=" * 70)

        bugs = self.analyze_code(module)

        if not bugs:
            print("\n✓ No bugs detected!")
            return

        print(f"\n✗ Found {len(bugs)} potential bug(s):\n")

        for i, bug in enumerate(bugs, 1):
            icon = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(bug['severity'], '⚪')

            print(f"{i}. {icon} [{bug['severity'].upper()}] {bug['type']}")
            print(f"   Location: {bug['location']}")
            print(f"   Issue: {bug['description']}")
            print(f"   Fix: {bug['suggestion']}\n")

        print("=" * 70)


# ---------------------------------------------------------
#   ATTENTION VALIDATOR
# ---------------------------------------------------------
class AttentionValidator:
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    # ======================================================
    def validate_attention_weights(self, attention_weights):
        sums = attention_weights.sum(dim=-1)
        if torch.allclose(sums, torch.ones_like(sums), atol=self.tolerance):
            return True, "Attention weights valid"
        return False, "Attention weights do not sum to 1"

    # ======================================================
    def validate_cache_shapes(self, cache, expected_seq_len):
        if cache["key"].shape[1] != expected_seq_len:
            return False, "Key cache seq_len mismatch"
        if cache["value"].shape[1] != expected_seq_len:
            return False, "Value cache seq_len mismatch"
        return True, "Cache shapes valid"

    # ======================================================
    def validate_output_shape(self, output, query):
        if output.shape == query.shape:
            return True, "Output shape valid"
        return False, "Output does not match query shape"

    # ======================================================
    def validate_causal_mask(self, attn, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        masked_vals = attn[..., mask]
        if torch.all(masked_vals < 1e-6):
            return True, "Causal mask valid"
        return False, "Causal mask violated"


# ---------------------------------------------------------
#   MAIN
# ---------------------------------------------------------
def main():
    print("AI Attention Debugger Ready.")


if __name__ == "__main__":
    main()
