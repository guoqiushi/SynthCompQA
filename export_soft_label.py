import os
import json
import math
import argparse
from typing import List, Dict, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor

# Try several possible model classes for compatibility
_MODEL_LOAD_SUCCESS = False
MODEL_CLASS = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    MODEL_CLASS = Qwen2_5_VLForConditionalGeneration
    _MODEL_LOAD_SUCCESS = True
except Exception:
    pass

if not _MODEL_LOAD_SUCCESS:
    try:
        from transformers import Qwen2VLForConditionalGeneration
        MODEL_CLASS = Qwen2VLForConditionalGeneration
        _MODEL_LOAD_SUCCESS = True
    except Exception:
        pass

if not _MODEL_LOAD_SUCCESS:
    try:
        from transformers import AutoModelForVision2Seq
        MODEL_CLASS = AutoModelForVision2Seq
        _MODEL_LOAD_SUCCESS = True
    except Exception:
        pass

if not _MODEL_LOAD_SUCCESS:
    raise ImportError(
        "Cannot find a compatible Qwen VL model class. "
        "Please upgrade transformers, or install a version supporting Qwen2-VL / Qwen2.5-VL."
    )


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export teacher soft labels from a fine-tuned Qwen VL model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path of the fine-tuned Qwen VL model.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Directory containing images. Recursive scan is supported.",
    )
    parser.add_argument(
        "--input_list",
        type=str,
        default="",
        help="Optional txt/jsonl file containing image paths. One path per line, or jsonl with key=image.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to save exported teacher labels.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "You are an image composition quality assessor. "
            "Evaluate only semantic plausibility and spatial consistency. "
            "Ignore fine-grained aesthetics. "
            "Classify the image into one of four quality levels: low, medium, good, perfect."
        ),
        help="Prompt used for teacher inference.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="low,medium,good,perfect",
        help="Comma-separated class labels in ascending quality order.",
    )
    parser.add_argument(
        "--anchors",
        type=str,
        default="0.1,0.4,0.7,0.95",
        help="Comma-separated anchor scores for labels, same order as --labels.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for images. For Qwen-VL scoring, batch_size=1 is the safest.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda / cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Only process first N images. -1 means all.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already exported images if output_jsonl exists.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def collect_images_from_dir(input_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in VALID_EXTS:
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


def collect_images_from_list(input_list: str) -> List[str]:
    paths = []
    if input_list.endswith(".jsonl"):
        with open(input_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "image" in obj:
                    paths.append(obj["image"])
    else:
        with open(input_list, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p:
                    paths.append(p)
    return paths


def load_image_paths(input_dir: str, input_list: str, max_samples: int) -> List[str]:
    all_paths = []
    if input_dir:
        all_paths.extend(collect_images_from_dir(input_dir))
    if input_list:
        all_paths.extend(collect_images_from_list(input_list))

    # deduplicate while preserving order
    seen = set()
    uniq = []
    for p in all_paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    if max_samples > 0:
        uniq = uniq[:max_samples]

    return uniq


def load_done_set(output_jsonl: str) -> set:
    done = set()
    if not os.path.exists(output_jsonl):
        return done
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "image" in obj:
                    done.add(obj["image"])
            except Exception:
                continue
    return done


def build_prompt_messages(prompt_text: str):
    """
    Prompt-only messages, with add_generation_prompt=True later.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def build_full_messages(prompt_text: str, candidate_label: str):
    """
    Prompt + candidate answer messages.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": candidate_label}
            ],
        },
    ]


def safe_apply_chat_template(processor, messages, add_generation_prompt: bool):
    """
    Try multiple compatible ways to build text prompt.
    """
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except TypeError:
        # older versions may not support add_generation_prompt exactly the same way
        return processor.apply_chat_template(messages, tokenize=False)


@torch.no_grad()
def encode_prompt_only(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    device: torch.device,
):
    """
    Encode prompt-only sequence to get prompt length.
    """
    prompt_messages = build_prompt_messages(prompt_text)
    prompt_str = safe_apply_chat_template(
        processor, prompt_messages, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt_str],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


@torch.no_grad()
def score_candidate_label(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    candidate_label: str,
    prompt_len: int,
    device: torch.device,
) -> float:
    """
    Compute conditional log-probability of candidate_label given image+prompt.

    We construct:
        [prompt + assistant_start] + candidate_label
    then sum token log-probs over candidate tokens.
    """
    full_messages = build_full_messages(prompt_text, candidate_label)
    full_str = safe_apply_chat_template(
        processor, full_messages, add_generation_prompt=False
    )

    full_inputs = processor(
        text=[full_str],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}

    input_ids = full_inputs["input_ids"]  # [1, L]
    full_len = input_ids.shape[1]

    if full_len <= prompt_len:
        # fallback: extremely unlikely, but keep safe
        return -1e9

    outputs = model(**full_inputs)
    logits = outputs.logits  # [1, L, V]

    # candidate token ids
    candidate_ids = input_ids[:, prompt_len:full_len]  # [1, T]
    # logits that predict those candidate tokens
    pred_logits = logits[:, prompt_len - 1 : full_len - 1, :]  # [1, T, V]

    log_probs = torch.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, candidate_ids.unsqueeze(-1)).squeeze(-1)

    seq_log_prob = token_log_probs.sum().item()
    return seq_log_prob


def softmax_from_log_scores(log_scores: List[float]) -> List[float]:
    m = max(log_scores)
    exps = [math.exp(x - m) for x in log_scores]
    s = sum(exps)
    return [x / s for x in exps]


def compute_teacher_score(probs: List[float], anchors: List[float]) -> float:
    return float(sum(p * a for p, a in zip(probs, anchors)))


def load_model_and_processor(model_path: str, device: str, dtype_str: str):
    dtype = get_torch_dtype(dtype_str)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    try:
        model = MODEL_CLASS.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            model = model.to(device)
    except Exception:
        model = MODEL_CLASS.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

    model.eval()
    return model, processor


def export_labels(
    model,
    processor,
    image_paths: List[str],
    output_jsonl: str,
    prompt_text: str,
    labels: List[str],
    anchors: List[float],
    device: str,
    resume: bool,
):
    done_set = load_done_set(output_jsonl) if resume else set()

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    fout = open(output_jsonl, "a" if resume else "w", encoding="utf-8")

    for img_path in tqdm(image_paths, desc="Exporting teacher labels"):
        if resume and img_path in done_set:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open image: {img_path} | {e}")
            continue

        try:
            prompt_inputs = encode_prompt_only(
                model=model,
                processor=processor,
                image=image,
                prompt_text=prompt_text,
                device=torch.device(device),
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            label_log_scores = []
            for lb in labels:
                score = score_candidate_label(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt_text=prompt_text,
                    candidate_label=lb,
                    prompt_len=prompt_len,
                    device=torch.device(device),
                )
                label_log_scores.append(score)

            teacher_probs = softmax_from_log_scores(label_log_scores)
            best_idx = max(range(len(labels)), key=lambda i: teacher_probs[i])

            teacher_label = labels[best_idx]
            teacher_conf = float(max(teacher_probs))
            teacher_score = compute_teacher_score(teacher_probs, anchors)

            result = {
                "image": img_path,
                "teacher_probs": [round(float(x), 8) for x in teacher_probs],
                "teacher_label": teacher_label,
                "teacher_score": round(float(teacher_score), 8),
                "teacher_conf": round(float(teacher_conf), 8),
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARN] OOM on image: {img_path}, skip.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                print(f"[WARN] Runtime error on {img_path}: {e}")
                continue
        except Exception as e:
            print(f"[WARN] Failed on image: {img_path} | {e}")
            continue

    fout.close()


def main():
    args = parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    anchors = [float(x.strip()) for x in args.anchors.split(",") if x.strip()]

    if len(labels) != len(anchors):
        raise ValueError("--labels and --anchors must have the same length.")

    if not args.input_dir and not args.input_list:
        raise ValueError("Please provide at least one of --input_dir or --input_list.")

    image_paths = load_image_paths(args.input_dir, args.input_list, args.max_samples)
    if len(image_paths) == 0:
        raise ValueError("No images found.")

    print(f"Found {len(image_paths)} images.")
    print(f"Loading model from: {args.model_path}")

    model, processor = load_model_and_processor(
        model_path=args.model_path,
        device=args.device,
        dtype_str=args.dtype,
    )

    export_labels(
        model=model,
        processor=processor,
        image_paths=image_paths,
        output_jsonl=args.output_jsonl,
        prompt_text=args.prompt,
        labels=labels,
        anchors=anchors,
        device=args.device,
        resume=args.resume,
    )

    print(f"Done. Results saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
