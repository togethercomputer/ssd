import re
from transformers import AutoTokenizer


# Infer model family based on model path name
def infer_model_family(model_path: str) -> str:
        """Infer if model is Llama or Qwen based on path name."""
        model_path_lower = model_path.lower()
        if "llama" in model_path_lower:
            return "llama"
        elif "qwen" in model_path_lower:
            return "qwen"
        else:
            return "unknown"


def decode_tokens(token_ids: list[int], tokenizer: AutoTokenizer) -> list[str]:
    decoded = []
    for token in token_ids:
        try:
            text = tokenizer.decode([token], skip_special_tokens=False)
            decoded.append(text)
        except Exception:
            decoded.append(f"<token_id:{token}>")
    return decoded


def compress_neg_ones_and_zeros(long_str: str) -> str:
    sub1 = re.sub(r'-1(?:, -1){2,}', '-1, ..., -1', long_str)
    sub2 = re.sub(r'0(?:, 0){2,}', '0, ..., 0', sub1)
    return sub2
