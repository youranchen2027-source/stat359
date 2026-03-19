import torch
import argparse
import os
import json

from bpe_tokenizer import BPETokenizer
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM


def load_tokenizer(tokenizer_path):
    return BPETokenizer.load(tokenizer_path)


def load_model_and_tokenizer(model_path, tokenizer_path, device):
    tokenizer = load_tokenizer(tokenizer_path)

    # Same pattern as the original script:
    # load args.json from the checkpoint directory if available
    config_path = os.path.join(os.path.dirname(model_path), "args.json")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            train_args = json.load(f)

        config = TinyStoriesConfig(
            vocab_size=len(tokenizer.token2id),
            hidden_size=train_args.get("hidden_size", 256),
            num_hidden_layers=train_args.get("num_layers", 4),
            num_attention_heads=train_args.get("num_heads", 8),
            intermediate_size=train_args.get("intermediate_size", 1024),
            hidden_dropout_prob=train_args.get("dropout", 0.1),
            attention_probs_dropout_prob=train_args.get("dropout", 0.1),
            max_position_embeddings=train_args.get("max_seq_len", 512),
            window_size=train_args.get("window_size", 256),
        )
    else:
        config = TinyStoriesConfig(vocab_size=len(tokenizer.token2id))

    model = TinyStoriesForCausalLM(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer


def read_test_prompts(test_prompts_path, id_key="id", prompt_key="prompt"):
    rows = []
    with open(test_prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append({
                "id": str(item[id_key]),
                "prompt": item[prompt_key],
            })
    return rows


def build_input_prompt(prompt, use_chat_format=False):
    if use_chat_format:
        return f"<user> {prompt} <assistant>"
    return prompt


def strip_generated_text(output_text, raw_prompt, chat_prompt=None, strip_prompt=True):
    if not strip_prompt:
        return output_text.strip()

    text = output_text.strip()

    # Try stripping the chat-formatted prefix first
    if chat_prompt is not None and text.startswith(chat_prompt):
        text = text[len(chat_prompt):].strip()

    # Fallback: strip the raw prompt if it appears at the start
    if text.startswith(raw_prompt):
        text = text[len(raw_prompt):].strip()

    # Remove leading assistant tag if the model repeats it
    if text.startswith("<assistant>"):
        text = text[len("<assistant>"):].strip()

    return text.strip()


def generate_one(
    model,
    tokenizer,
    prompt,
    device,
    max_length=120,
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    strip_prompt=True,
    use_chat_format=False,
):
    model_input_text = build_input_prompt(prompt, use_chat_format=use_chat_format)

    input_ids = torch.tensor(
        [tokenizer.encode(model_input_text, add_special_tokens=True)],
        dtype=torch.long
    ).to(device)

    eos_token_id = tokenizer.token2id.get("<eos>", None)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0].tolist())
    cleaned_text = strip_generated_text(
        output_text=output_text,
        raw_prompt=prompt,
        chat_prompt=model_input_text if use_chat_format else None,
        strip_prompt=strip_prompt,
    )
    return cleaned_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate batched outputs from test_prompts.jsonl using a trained TinyStories model."
    )

    # Keep the original-style arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="tinystories_model/best_model.pth",
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="bpe_tokenizer_tinystories.pkl",
        help="Path to the BPE tokenizer"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=120,
        help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use"
    )

    # New minimal additions for batched evaluation generation
    parser.add_argument(
        "--test_prompts",
        type=str,
        required=True,
        help="Path to test_prompts.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save generated outputs JSONL"
    )
    parser.add_argument(
        "--id_key",
        type=str,
        default="id",
        help="ID key in test prompts JSONL"
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="prompt",
        help="Prompt key in test prompts JSONL"
    )
    parser.add_argument(
        "--strip_prompt",
        action="store_true",
        help="Strip prompt/chat prefix from decoded output if present"
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="Wrap prompt as '<user> ... <assistant>' before generation"
    )

    args = parser.parse_args()

    # Device selection: original script supports auto/cpu/cuda/mps,
    # though its default auto path picks cuda if available else cpu. :contentReference[oaicite:1]{index=1}
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
    )

    prompt_rows = read_test_prompts(
        test_prompts_path=args.test_prompts,
        id_key=args.id_key,
        prompt_key=args.prompt_key,
    )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for idx, row in enumerate(prompt_rows, start=1):
            response = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=row["prompt"],
                device=device,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                strip_prompt=args.strip_prompt,
                use_chat_format=args.use_chat_format,
            )

            fout.write(json.dumps({
                "id": row["id"],
                "response": response,
            }, ensure_ascii=False) + "\n")

            if idx % 10 == 0 or idx == len(prompt_rows):
                print(f"Generated {idx}/{len(prompt_rows)}")

    print(f"Saved outputs to: {args.output_file}")


if __name__ == "__main__":
    main()