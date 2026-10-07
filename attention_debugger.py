"""
AI Attention Debugger - BONUS Challenge (+10%)

Fully implemented & debugged version.
"""

import torch
from typing import Any, List, Dict
import ast
import inspect
import re
import argparse
import importlib.util
import sys
from pathlib import Path


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
    def analyze_code(self, module) -> List[Dict[str, Any]]:
        bugs = []
        if not hasattr(module, "KVCachedMultiHeadAttention"):
            bugs.append({
                'type': 'missing_attention_class',
                'severity': 'critical',
                'location': 'module',
                'description': "Required class 'KVCachedMultiHeadAttention' was not found",
                'suggestion': "Define KVCachedMultiHeadAttention in the module"
            })
            self.detected_bugs = bugs
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
        self.detected_bugs = bugs
        return bugs

    # ======================================================
    # SCALING FACTOR CHECK
    # ======================================================
    def check_scaling_factor(self, model_class):
        src, _ = get_source_ast(model_class)

        compact_source = re.sub(r"\s+", "", src)
        if "/head_dim" in compact_source or "/self.head_dim" in compact_source or "/d_k" in compact_source:
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
        src, _ = get_source_ast(model_class)

        if "softmax" in src and re.search(r"dim\s*=\s*-2\b", src):
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
        _, tree = get_source_ast(model_class)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "cat"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                continue
            dim = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "dim"),
                node.args[1] if len(node.args) > 1 else ast.Constant(value=0)
            )
            if not isinstance(dim, ast.Constant) or dim.value != 1:
                return {
                    'type': 'cache_concat',
                    'severity': 'high',
                    'location': 'cache update',
                    'description': 'Flattened key/value caches must be concatenated along the sequence dimension',
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
        src, tree = get_source_ast(model_class)
        bugs = []

        def integer_value(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return node.value
            if (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, int)
            ):
                return -node.operand.value
            return None

        transposes_keys = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "transpose"
            and len(node.args) >= 2
            and tuple(integer_value(arg) for arg in node.args[:2]) in ((-2, -1), (2, 3))
            for node in ast.walk(tree)
        )
        if "matmul" in src and not transposes_keys:
            bugs.append({
                'type': 'missing_transpose',
                'severity': 'medium',
                'location': 'QK^T matmul',
                'description': 'Key/value tensors not transposed',
                'suggestion': 'Ensure K = K.transpose(2, 3)'
            })

        if "reshape" in src and "num_heads" in src and "head_dim" not in src and "d_head" not in src:
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
        if cache is None or cache.get("key") is None or cache.get("value") is None:
            return False, "Key/value cache is missing"
        if cache["key"].dim() != 3 or cache["value"].dim() != 3:
            return False, "Cache tensors must have shape (batch, seq_len, d_model)"
        if cache["key"].shape != cache["value"].shape:
            return False, "Key and value cache shapes do not match"
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
    def validate_causal_mask(self, attn, seq_len, cache_len=0):
        if attn.dim() < 2 or attn.shape[-2] != seq_len:
            return False, "Attention weights query length mismatch"
        mask = torch.triu(
            torch.ones(seq_len, attn.shape[-1], device=attn.device, dtype=torch.bool),
            diagonal=cache_len + 1
        )
        masked_vals = attn[..., mask]
        if torch.all(masked_vals < 1e-6):
            return True, "Causal mask valid"
        return False, "Causal mask violated"


# ---------------------------------------------------------
#   MAIN
# ---------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Analyze a KV-cached attention implementation")
    parser.add_argument(
        "--file",
        default=str(Path(__file__).resolve().with_name("kv_attention.py")),
        help="Python module to analyze (default: kv_attention.py beside this script)"
    )
    args = parser.parse_args()

    module_path = Path(args.file)
    if not module_path.is_file():
        print(f"Error: Python module '{module_path}' does not exist", file=sys.stderr)
        return 2

    spec = importlib.util.spec_from_file_location("attention_module", module_path)
    if spec is None or spec.loader is None:
        print(f"Error: Unable to load Python module '{module_path}'", file=sys.stderr)
        return 2

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(spec.name, None)
        print(f"Error loading Python module: {error}", file=sys.stderr)
        return 2

    detector = AttentionBugDetector()
    bugs = detector.analyze_code(module)
    detector.run_analysis(module)
    return 1 if bugs else 0


if __name__ == "__main__":
    sys.exit(main())
