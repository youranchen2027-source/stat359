# TinyStories LLM: Personality Engineering for Social Interaction Style
The project fine-tunes the base model using instruction-style dialogue data and evaluate the resulting models using a multi-dimensional framework.

### Environment Setup
Install dependencies using Poetry:
```bash
poetry install
```

### Step 1: Train the Tokenizer
Use the provided script to train a tokenizer:
```bash
poetry run python instructor/final_project/tinystories_llm/train_bpe_tokenizer_hf.py
```

### Step 2: Train the base Model
Train the base model using:
```bash
poetry run python instructor/final_project/tinystories_llm/train_tinystories_model.py --amp
```

### Step 3: Persona Dataset Curation
Generate the initial 2,000 user prompts
```bash
poetry run python generate_user_prompts.py \
  --output_file user_prompts.txt \
  --target_size 2000 \
  --seed 42
```

Generate 10,000 samples for every persona
```bash
poetry run python build_persona_dataset.py \
  --input_file user_prompts.txt \
  --output_dir persona_datasets \
  --target_size 10000 \
  --valid_ratio 0.1 \
  --seed 42 \
  --max_replies_per_prompt 12
```

### Step 4: Chat with the persona models
Interact with the persona model:
```bash
poetry run python instructor/final_project/tinystories_llm/chat_with_tinystories_model.py \
  --model_path friendly_chat_model/final_model.pth
```

### Step 5: Preparation for Evaluation
Generate text with the trained models to evaluate whether the generated text is meaningful.
```bash
poetry run python generate_model_outputs.py \
  --model_path friendly_chat_model/final_model.pth \
  --tokenizer_path bpe_tokenizer_tinystories.pkl \
  --test_prompts test_prompts.jsonl \
  --output_file model_outputs/friendly_outputs.jsonl \
  --max_length 120 \
  --temperature 0.9 \
  --top_p 0.9 \
  --strip_prompt \
  --use_chat_format
```

### Step 6: Evaluation
Install dependency for OpenAI api:
```bash
pip install openai
export OPENAI_API_KEY=your_own_key
```

Generate evaluation results
```bash
poetry run python evaluate_persona_models.py \
  --test_prompts test_prompts.jsonl \
  --outputs_dir model_outputs \
  --output_dir evaluation_results \
  --use_gpt_judge \
  --judge_model gpt-4o-mini \
  --max_judge_samples 100
```






  
