import argparse
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# =========================================================
# Argument parsing
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate persona-engineered LLM outputs."
    )
    parser.add_argument(
        "--test_prompts",
        type=str,
        required=True,
        help="Path to unified test prompts JSONL."
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        required=True,
        help="Directory containing persona output JSONL files."
    )
    parser.add_argument(
        "--personas",
        type=str,
        nargs="+",
        default=["friendly", "robot", "sarcastic"],
        help="Persona names to evaluate."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for automatic judging."
    )
    parser.add_argument(
        "--use_gpt_judge",
        action="store_true",
        help="Enable GPT-based judging."
    )
    parser.add_argument(
        "--max_judge_samples",
        type=int,
        default=None,
        help="Maximum number of samples per persona to judge with GPT."
    )
    parser.add_argument(
        "--response_key",
        type=str,
        default="response",
        help="Field name for response text in model output JSONL."
    )
    parser.add_argument(
        "--id_key",
        type=str,
        default="id",
        help="Field name for id in both prompt/output JSONL."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()


# =========================================================
# Data classes
# =========================================================

@dataclass
class PromptItem:
    sample_id: str
    prompt: str
    category: str
    reference: str = ""


@dataclass
class ResponseItem:
    sample_id: str
    response: str


# =========================================================
# Persona style lexicons
# =========================================================

PERSONA_DESCRIPTIONS = {
    "friendly": "Warm, supportive, empathetic, encouraging, emotionally gentle.",
    "robot": "Structured, concise, logical, emotionally neutral, step-by-step.",
    "sarcastic": "Lightly sarcastic or dryly humorous, but still helpful and not abusive.",
}

PERSONA_KEYWORDS = {
    "friendly": [
        "i understand", "you are not alone", "you're not alone", "it's okay",
        "it is okay", "take a breath", "one step at a time", "be kind to yourself",
        "you can do this", "that sounds hard", "i'm sorry", "small step",
        "you are doing better", "gently", "you deserve patience"
    ],
    "robot": [
        "step", "analysis", "recommendation", "conclusion", "objective",
        "define", "identify", "evaluate", "structured", "systematic",
        "process", "plan", "priority", "procedure", "assessment"
    ],
    "sarcastic": [
        "well", "classic", "not ideal", "dramatic", "human problems",
        "spiraling", "panic", "tragedy", "messy", "rough", "anyway",
        "not the end of the world", "welcome to", "emotional theater"
    ],
}

TONE_MARKERS = {
    "friendly": {
        "empathy": [
            "i understand", "that sounds hard", "i'm sorry", "you are not alone",
            "it's okay", "it is okay", "be kind to yourself", "take a breath"
        ],
        "encouragement": [
            "you can", "small step", "one step at a time", "you can do this",
            "you are doing better", "keep going", "you've got this"
        ],
    },
    "robot": {
        "structure": [
            "1.", "2.", "3.", "step", "first", "second", "third",
            "recommendation", "conclusion", "analysis"
        ],
        "objectivity": [
            "objective", "identify", "evaluate", "define", "process", "plan",
            "assessment", "procedure", "systematic"
        ],
    },
    "sarcastic": {
        "sarcasm": [
            "well", "classic", "not ideal", "dramatic", "spiraling",
            "panic", "tragedy", "human problems", "anyway", "messy"
        ],
        "dry_humor": [
            "rough", "not the end of the world", "welcome to", "emotional theater"
        ],
    },
}


# =========================================================
# File I/O
# =========================================================

def load_test_prompts(path: str, id_key: str) -> Dict[str, PromptItem]:
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid = str(row[id_key])
            items[sid] = PromptItem(
                sample_id=sid,
                prompt=row["prompt"],
                category=row.get("category", "unknown"),
                reference=row.get("reference", "")
            )
    return items


def load_responses(path: str, id_key: str, response_key: str) -> Dict[str, ResponseItem]:
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid = str(row[id_key])
            items[sid] = ResponseItem(
                sample_id=sid,
                response=row[response_key]
            )
    return items


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(rows: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_markdown(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =========================================================
# Basic text processing
# =========================================================

def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_std(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def count_phrase_hits(text: str, phrase_list: List[str]) -> int:
    text_lower = text.lower()
    return sum(1 for p in phrase_list if p.lower() in text_lower)


# =========================================================
# Voice characteristics
# =========================================================

def compute_voice_metrics(persona: str, responses: List[str]) -> Dict[str, float]:
    token_counts = []
    sentence_counts = []
    avg_sentence_lengths = []
    question_marks = []
    exclamation_marks = []
    keyword_hits = []
    keyword_coverage = []
    tone_stats = defaultdict(list)

    persona_keywords = PERSONA_KEYWORDS.get(persona, [])
    tone_groups = TONE_MARKERS.get(persona, {})

    for response in responses:
        tokens = tokenize(response)
        sentences = split_sentences(response)

        token_counts.append(len(tokens))
        sentence_counts.append(len(sentences))

        if sentences:
            sent_lens = [len(tokenize(s)) for s in sentences]
            avg_sentence_lengths.append(average(sent_lens))
        else:
            avg_sentence_lengths.append(0.0)

        question_marks.append(response.count("?"))
        exclamation_marks.append(response.count("!"))

        hits = count_phrase_hits(response, persona_keywords)
        keyword_hits.append(hits)
        keyword_coverage.append(1.0 if hits > 0 else 0.0)

        for tone_name, phrases in tone_groups.items():
            hit = count_phrase_hits(response, phrases)
            tone_stats[tone_name].append(hit)

    metrics = {
        "avg_tokens": round(average(token_counts), 4),
        "std_tokens": round(safe_std(token_counts), 4),
        "avg_sentences": round(average(sentence_counts), 4),
        "avg_sentence_length": round(average(avg_sentence_lengths), 4),
        "avg_question_marks": round(average(question_marks), 4),
        "avg_exclamation_marks": round(average(exclamation_marks), 4),
        "avg_keyword_hits": round(average(keyword_hits), 4),
        "keyword_coverage_rate": round(average(keyword_coverage), 4),
    }

    for tone_name, values in tone_stats.items():
        metrics[f"{tone_name}_avg_hits"] = round(average(values), 4)
        metrics[f"{tone_name}_coverage_rate"] = round(
            average([1.0 if v > 0 else 0.0 for v in values]), 4
        )

    return metrics


def compute_lexical_metrics(responses: List[str]) -> Dict[str, float]:
    all_tokens = []
    per_response_ttr = []
    repeated_bigram_counts = []

    for response in responses:
        tokens = tokenize(response)
        all_tokens.extend(tokens)

        if tokens:
            per_response_ttr.append(len(set(tokens)) / len(tokens))
        else:
            per_response_ttr.append(0.0)

        bigrams = list(zip(tokens, tokens[1:]))
        if bigrams:
            c = Counter(bigrams)
            repeated = sum(v for v in c.values() if v > 1)
            repeated_bigram_counts.append(repeated)
        else:
            repeated_bigram_counts.append(0)

    corpus_ttr = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0

    return {
        "avg_type_token_ratio": round(average(per_response_ttr), 4),
        "corpus_type_token_ratio": round(corpus_ttr, 4),
        "avg_repeated_bigram_count": round(average(repeated_bigram_counts), 4),
    }


def text_to_counter(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine_counter(c1: Counter, c2: Counter) -> float:
    common = set(c1.keys()) & set(c2.keys())
    dot = sum(c1[k] * c2[k] for k in common)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def compute_style_consistency_proxy(responses: List[str], max_pairs: int = 500) -> Dict[str, float]:
    counters = [text_to_counter(r) for r in responses]
    sims = []
    n = len(counters)

    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_counter(counters[i], counters[j]))
            pair_count += 1
            if pair_count >= max_pairs:
                break
        if pair_count >= max_pairs:
            break

    return {
        "style_consistency_proxy_mean": round(average(sims), 4),
        "style_consistency_proxy_std": round(safe_std(sims), 4),
    }


def evaluate_voice(persona: str, responses: List[str]) -> Dict[str, float]:
    metrics = {}
    metrics.update(compute_voice_metrics(persona, responses))
    metrics.update(compute_lexical_metrics(responses))
    metrics.update(compute_style_consistency_proxy(responses))
    return metrics


# =========================================================
# GPT Judge helpers
# =========================================================

def get_openai_client():
    if OpenAI is None:
        raise ImportError("openai package is not installed. Run: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def parse_judge_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def judge_persona_consistency(
    client,
    judge_model: str,
    persona: str,
    prompt: str,
    response: str,
) -> dict:
    system_prompt = (
        "You are an impartial evaluator for persona-conditioned language model outputs. "
        "Return ONLY valid JSON."
    )

    user_prompt = f"""
Evaluate whether the model response matches the target persona.

Target persona: {persona}
Persona description: {PERSONA_DESCRIPTIONS.get(persona, persona)}

User prompt:
{prompt}

Model response:
{response}

Score the response on:
1. persona_match (1-5): how well it matches the target persona
2. naturalness (1-5): whether the response sounds natural
3. consistency (1-5): whether the style is internally consistent

Also provide:
- short_reason
- failure_type: one of ["none", "too_neutral", "wrong_persona", "forced_style", "unsafe_tone"]

Return JSON with keys:
{{
  "persona_match": int,
  "naturalness": int,
  "consistency": int,
  "short_reason": str,
  "failure_type": str
}}
""".strip()

    completion = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    text = completion.choices[0].message.content
    return parse_judge_json(text)


def judge_task_quality(
    client,
    judge_model: str,
    prompt: str,
    response: str,
    reference: str = "",
) -> dict:
    system_prompt = (
        "You are an impartial evaluator for language model response quality. "
        "Return ONLY valid JSON."
    )

    user_prompt = f"""
Evaluate the quality of the model response.

User prompt:
{prompt}

Model response:
{response}

Reference answer (may be empty):
{reference}

Score the response on:
1. helpfulness (1-5): does it address the user's need?
2. fluency (1-5): is it grammatically fluent and readable?
3. correctness (1-5): factual or task correctness, using the reference if available

Also provide:
- short_reason
- error_type: one of ["none", "incorrect", "vague", "incomplete", "off_topic", "hallucinated"]

Return JSON with keys:
{{
  "helpfulness": int,
  "fluency": int,
  "correctness": int,
  "short_reason": str,
  "error_type": str
}}
""".strip()

    completion = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    text = completion.choices[0].message.content
    return parse_judge_json(text)


# =========================================================
# Evaluation pipelines
# =========================================================

def maybe_sample_ids(ids: List[str], max_samples: Optional[int], seed: int) -> List[str]:
    if max_samples is None or len(ids) <= max_samples:
        return ids
    rng = random.Random(seed)
    ids_copy = list(ids)
    rng.shuffle(ids_copy)
    return ids_copy[:max_samples]


def evaluate_persona_consistency_with_gpt(
    client,
    judge_model: str,
    persona: str,
    prompt_items: Dict[str, PromptItem],
    response_items: Dict[str, ResponseItem],
    max_samples: Optional[int],
    seed: int,
) -> Tuple[Dict[str, object], List[dict]]:
    sample_ids = sorted(set(prompt_items.keys()) & set(response_items.keys()))
    sample_ids = maybe_sample_ids(sample_ids, max_samples, seed)

    rows = []
    persona_match_scores = []
    naturalness_scores = []
    consistency_scores = []
    failure_counter = Counter()

    for sid in sample_ids:
        prompt = prompt_items[sid].prompt
        response = response_items[sid].response

        result = judge_persona_consistency(
            client=client,
            judge_model=judge_model,
            persona=persona,
            prompt=prompt,
            response=response,
        )

        row = {
            "id": sid,
            "persona": persona,
            "prompt": prompt,
            "response": response,
            **result,
        }
        rows.append(row)

        persona_match_scores.append(result["persona_match"])
        naturalness_scores.append(result["naturalness"])
        consistency_scores.append(result["consistency"])
        failure_counter[result["failure_type"]] += 1

    summary = {
        "persona_match_avg": round(average(persona_match_scores), 4),
        "naturalness_avg": round(average(naturalness_scores), 4),
        "consistency_avg": round(average(consistency_scores), 4),
        "failure_distribution": dict(failure_counter),
        "num_scored": len(rows),
    }

    return summary, rows


def evaluate_task_quality_with_gpt(
    client,
    judge_model: str,
    persona: str,
    prompt_items: Dict[str, PromptItem],
    response_items: Dict[str, ResponseItem],
    max_samples: Optional[int],
    seed: int,
) -> Tuple[Dict[str, object], List[dict]]:
    sample_ids = sorted(set(prompt_items.keys()) & set(response_items.keys()))
    sample_ids = maybe_sample_ids(sample_ids, max_samples, seed)

    rows = []
    helpfulness_scores = []
    fluency_scores = []
    correctness_scores = []
    error_counter = Counter()
    category_scores = defaultdict(lambda: {"helpfulness": [], "fluency": [], "correctness": []})

    for sid in sample_ids:
        item = prompt_items[sid]
        response = response_items[sid].response

        result = judge_task_quality(
            client=client,
            judge_model=judge_model,
            prompt=item.prompt,
            response=response,
            reference=item.reference,
        )

        row = {
            "id": sid,
            "persona": persona,
            "category": item.category,
            "prompt": item.prompt,
            "reference": item.reference,
            "response": response,
            **result,
        }
        rows.append(row)

        helpfulness_scores.append(result["helpfulness"])
        fluency_scores.append(result["fluency"])
        correctness_scores.append(result["correctness"])
        error_counter[result["error_type"]] += 1

        category_scores[item.category]["helpfulness"].append(result["helpfulness"])
        category_scores[item.category]["fluency"].append(result["fluency"])
        category_scores[item.category]["correctness"].append(result["correctness"])

    category_summary = {}
    for cat, vals in category_scores.items():
        category_summary[cat] = {
            "helpfulness_avg": round(average(vals["helpfulness"]), 4),
            "fluency_avg": round(average(vals["fluency"]), 4),
            "correctness_avg": round(average(vals["correctness"]), 4),
            "count": len(vals["helpfulness"]),
        }

    summary = {
        "helpfulness_avg": round(average(helpfulness_scores), 4),
        "fluency_avg": round(average(fluency_scores), 4),
        "correctness_avg": round(average(correctness_scores), 4),
        "error_distribution": dict(error_counter),
        "category_breakdown": category_summary,
        "num_scored": len(rows),
    }

    return summary, rows


# =========================================================
# Markdown report
# =========================================================

def build_markdown_report(overall_summary: Dict[str, dict]) -> str:
    lines = []
    lines.append("# Persona Evaluation Report")
    lines.append("")

    for persona, report in overall_summary.items():
        lines.append(f"## Persona: {persona}")
        lines.append("")
        lines.append(f"- Aligned samples: {report.get('num_aligned_samples', 0)}")
        lines.append("")

        voice = report.get("voice_characteristics", {})
        lines.append("### Voice Characteristics")
        for k, v in voice.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

        consistency = report.get("persona_consistency", {})
        lines.append("### Persona Consistency")
        for k, v in consistency.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

        quality = report.get("task_quality_preservation", {})
        lines.append("### Task Quality Preservation")
        for k, v in quality.items():
            if k == "category_breakdown":
                lines.append("- category_breakdown:")
                for cat, cat_vals in v.items():
                    lines.append(f"  - {cat}: {cat_vals}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")

    return "\n".join(lines)


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()
    random.seed(args.random_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_items = load_test_prompts(args.test_prompts, args.id_key)

    client = None
    if args.use_gpt_judge:
        client = get_openai_client()

    overall_summary = {}

    for idx, persona in enumerate(args.personas):
        persona_output_path = Path(args.outputs_dir) / f"{persona}_outputs.jsonl"
        if not persona_output_path.exists():
            raise FileNotFoundError(f"Missing output file: {persona_output_path}")

        response_items = load_responses(
            str(persona_output_path),
            id_key=args.id_key,
            response_key=args.response_key
        )

        aligned_ids = sorted(set(prompt_items.keys()) & set(response_items.keys()))
        responses = [response_items[sid].response for sid in aligned_ids]

        persona_report = {
            "persona": persona,
            "num_aligned_samples": len(aligned_ids),
        }

        # Voice characteristics
        voice_summary = evaluate_voice(persona, responses)
        persona_report["voice_characteristics"] = voice_summary

        # Persona consistency
        if args.use_gpt_judge:
            consistency_summary, consistency_rows = evaluate_persona_consistency_with_gpt(
                client=client,
                judge_model=args.judge_model,
                persona=persona,
                prompt_items=prompt_items,
                response_items=response_items,
                max_samples=args.max_judge_samples,
                seed=args.random_seed + idx * 100,
            )
            persona_report["persona_consistency"] = consistency_summary
            save_jsonl(
                consistency_rows,
                output_dir / f"{persona}_persona_consistency_details.jsonl"
            )
        else:
            persona_report["persona_consistency"] = {
                "note": "GPT judge disabled. Run with --use_gpt_judge."
            }

        # Task quality preservation
        if args.use_gpt_judge:
            quality_summary, quality_rows = evaluate_task_quality_with_gpt(
                client=client,
                judge_model=args.judge_model,
                persona=persona,
                prompt_items=prompt_items,
                response_items=response_items,
                max_samples=args.max_judge_samples,
                seed=args.random_seed + idx * 1000,
            )
            persona_report["task_quality_preservation"] = quality_summary
            save_jsonl(
                quality_rows,
                output_dir / f"{persona}_task_quality_details.jsonl"
            )
        else:
            persona_report["task_quality_preservation"] = {
                "note": "GPT judge disabled. Run with --use_gpt_judge."
            }

        overall_summary[persona] = persona_report
        save_json(persona_report, output_dir / f"{persona}_summary.json")

    save_json(overall_summary, output_dir / "overall_summary.json")
    report_md = build_markdown_report(overall_summary)
    save_markdown(report_md, output_dir / "report.md")

    print(f"Saved evaluation results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()