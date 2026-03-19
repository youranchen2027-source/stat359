import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate large-scale English user prompts.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="user_prompts.txt",
        help="Output txt file path."
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=2000,
        help="Target number of prompts to generate."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    return parser.parse_args()


# =========================================================
# Template building blocks
# =========================================================

EMOTIONS = [
    "sad", "stressed", "anxious", "overwhelmed", "upset", "lonely",
    "afraid", "worried", "exhausted", "frustrated", "hopeless", "burned out"
]

PROBLEMS = [
    "my exam", "my grades", "my study schedule", "my motivation",
    "my time management", "my confidence", "my friendships",
    "my social life", "my workload", "my future", "my habits", "my focus"
]

ACTIONS = [
    "improve", "fix", "manage", "deal with", "handle", "organize",
    "recover from", "stop", "reduce", "understand", "change"
]

TASKS = [
    "study more effectively",
    "stop procrastinating",
    "build better habits",
    "make a daily plan",
    "stay focused",
    "prepare for exams",
    "manage stress",
    "feel less anxious",
    "be more productive",
    "balance school and rest",
]

CONCEPTS = [
    "machine learning",
    "deep learning",
    "neural networks",
    "overfitting",
    "gradient descent",
    "large language models",
    "probability",
    "Bayes' theorem",
    "regularization",
    "transformers",
    "attention mechanisms",
    "diffusion models",
]

SOCIAL_SITUATIONS = [
    "talking to new people",
    "making friends",
    "joining a group conversation",
    "asking for help",
    "speaking in class",
    "dealing with awkward silence",
    "responding to criticism",
    "working in a team",
]

TIME_FRAMES = [
    "lately", "recently", "these days", "for the past few weeks", "this semester"
]

PERSONAL_STATES = [
    "I feel like I am not smart enough.",
    "I feel like I keep failing.",
    "I feel stuck.",
    "I feel tired all the time.",
    "I feel like nothing is working.",
    "I feel behind compared with everyone else.",
    "I feel unmotivated.",
    "I feel like I am wasting time.",
]

CHAT_TOPICS = [
    "success", "failure", "motivation", "discipline", "friendship",
    "stress", "learning", "confidence", "rest", "personal growth"
]

REQUEST_STYLES = [
    "Can you explain",
    "Can you help me understand",
    "Please explain",
    "Can you describe",
    "Can you summarize",
    "Tell me about",
]

OPENERS = [
    "How can I",
    "How do I",
    "What should I do to",
    "What is the best way to",
    "Do you have advice on how to",
]

REFLECTIONS = [
    "Why do I always lose motivation so quickly?",
    "Why is it so hard for me to stay consistent?",
    "Why do I get anxious before important things?",
    "Why do I compare myself to other people so much?",
    "Why do I procrastinate even when I care about the task?",
    "Why do I feel guilty when I rest?",
]

ENCOURAGEMENT_REQUESTS = [
    "Can you encourage me a little?",
    "Can you comfort me?",
    "Can you say something supportive?",
    "I need a little reassurance right now.",
    "Can you help me calm down?",
]

PLANNING_OBJECTS = [
    "a study plan",
    "a weekly schedule",
    "a simple routine",
    "a better workflow",
    "a realistic plan for tomorrow",
    "a step-by-step plan for this week",
]

EXAM_CONTEXTS = [
    "I failed my exam and feel terrible.",
    "I did badly on a test and now I feel discouraged.",
    "I studied hard but my exam score was still disappointing.",
    "My grades dropped and I do not know what to do next.",
    "I am scared because my exam is coming up soon.",
]

SELF_DOUBT_CONTEXTS = [
    "I do not think I am good enough.",
    "I feel like everyone else is doing better than me.",
    "I am starting to doubt myself.",
    "I feel like I am falling behind.",
    "I am worried that I am not capable enough.",
]

GENERAL_HELP_REQUESTS = [
    "Can you give me one practical suggestion?",
    "Can you give me three simple steps?",
    "Can you make the answer easy to understand?",
    "Can you keep the explanation short?",
    "Can you make it less overwhelming?",
]


# =========================================================
# Prompt generation functions
# =========================================================

def emotional_prompts():
    prompts = []
    for emotion in EMOTIONS:
        for problem in PROBLEMS:
            prompts.append(f"I have been feeling {emotion} about {problem} lately. What should I do?")
            prompts.append(f"I feel {emotion} because of {problem}. Can you help me?")
            prompts.append(f"I am {emotion} about {problem}. Do you have any advice?")
    return prompts


def study_productivity_prompts():
    prompts = []
    for opener in OPENERS:
        for task in TASKS:
            prompts.append(f"{opener} {task}?")
            prompts.append(f"{opener} {task} without feeling overwhelmed?")
            prompts.append(f"{opener} {task} more consistently?")
    return prompts


def explanation_prompts():
    prompts = []
    for style in REQUEST_STYLES:
        for concept in CONCEPTS:
            prompts.append(f"{style} {concept} in simple terms?")
            prompts.append(f"{style} {concept} like I am a beginner?")
            prompts.append(f"{style} why {concept} matters?")
    return prompts


def social_prompts():
    prompts = []
    for situation in SOCIAL_SITUATIONS:
        prompts.append(f"How can I get better at {situation}?")
        prompts.append(f"What should I do if I feel nervous about {situation}?")
        prompts.append(f"Can you give me advice about {situation}?")
    return prompts


def personal_state_prompts():
    prompts = []
    for state in PERSONAL_STATES:
        prompts.append(state)
        prompts.append(f"{state} What should I do?")
        prompts.append(f"{state} Can you help me think clearly?")
    return prompts


def reflection_prompts():
    return list(REFLECTIONS)


def encouragement_prompts():
    prompts = []
    for req in ENCOURAGEMENT_REQUESTS:
        prompts.append(req)
        prompts.append(f"{req} I have been under a lot of pressure lately.")
        prompts.append(f"{req} I do not feel very confident right now.")
    return prompts


def planning_prompts():
    prompts = []
    for obj in PLANNING_OBJECTS:
        prompts.append(f"Can you help me make {obj}?")
        prompts.append(f"How do I create {obj}?")
        prompts.append(f"What should I include in {obj}?")
    return prompts


def exam_prompts():
    prompts = []
    for ctx in EXAM_CONTEXTS:
        prompts.append(ctx)
        prompts.append(f"{ctx} Can you help me figure out what to do next?")
        prompts.append(f"{ctx} I need advice.")
    return prompts


def self_doubt_prompts():
    prompts = []
    for ctx in SELF_DOUBT_CONTEXTS:
        prompts.append(ctx)
        prompts.append(f"{ctx} Can you encourage me?")
        prompts.append(f"{ctx} What should I do?")
    return prompts


def casual_chat_prompts():
    prompts = []
    for topic in CHAT_TOPICS:
        prompts.append(f"What do you think about {topic}?")
        prompts.append(f"Can we talk about {topic}?")
        prompts.append(f"Why is {topic} important?")
        prompts.append(f"What is a healthy way to think about {topic}?")
    return prompts


def mixed_context_prompts():
    prompts = []
    for emotion in EMOTIONS:
        for task in TASKS[:6]:
            prompts.append(f"I feel {emotion}, and I still need to {task}. What should I do?")
            prompts.append(f"How can I {task} when I already feel {emotion}?")
    for timeframe in TIME_FRAMES:
        for problem in PROBLEMS[:8]:
            prompts.append(f"I have been struggling with {problem} {timeframe}. Can you help?")
    return prompts


def add_request_suffixes(base_prompts):
    suffixes = [
        "",
        " Please keep it simple.",
        " Please be direct.",
        " Please make it easy to understand.",
        " Please give practical advice.",
        " Please keep it short.",
        " Please be supportive.",
    ]
    expanded = []
    for p in base_prompts:
        for s in suffixes:
            expanded.append((p + s).strip())
    return expanded


def generate_all_prompt_candidates():
    prompt_groups = [
        emotional_prompts(),
        study_productivity_prompts(),
        explanation_prompts(),
        social_prompts(),
        personal_state_prompts(),
        reflection_prompts(),
        encouragement_prompts(),
        planning_prompts(),
        exam_prompts(),
        self_doubt_prompts(),
        casual_chat_prompts(),
        mixed_context_prompts(),
    ]

    all_prompts = []
    for group in prompt_groups:
        all_prompts.extend(group)

    all_prompts = add_request_suffixes(all_prompts)
    return all_prompts


def normalize_prompt(text: str) -> str:
    return " ".join(text.strip().split())


def deduplicate(prompts):
    seen = set()
    unique = []
    for p in prompts:
        p = normalize_prompt(p)
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    all_candidates = generate_all_prompt_candidates()
    unique_prompts = deduplicate(all_candidates)

    if len(unique_prompts) < args.target_size:
        print(
            f"Warning: only {len(unique_prompts)} unique prompts available, "
            f"which is smaller than target_size={args.target_size}."
        )
        selected = unique_prompts
    else:
        rng.shuffle(unique_prompts)
        selected = unique_prompts[:args.target_size]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for prompt in selected:
            f.write(prompt + "\n")

    print(f"Saved {len(selected)} prompts to {output_path.resolve()}")


if __name__ == "__main__":
    main()