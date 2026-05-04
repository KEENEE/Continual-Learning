from __future__ import annotations

import argparse
import openai
import data_generation_old.prompt
import json

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 128000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming chat completion (single request; no auto-continuation).",
    )
    parser.add_argument("--api_base", default="https://sr-aic-llm-proxy.lit.ovh")
    parser.add_argument("--api_key", default="sk-7hXVVhoPDmA16gQD9w1w7w")
    # parser.add_argument("--model", default="vertex_ai/claude-opus-4-6")
    parser.add_argument("--model", default="vertex_ai/gemini-3.1-pro-preview")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--out", default="out.txt")
    args = parser.parse_args()

    client = openai.OpenAI(api_key=args.api_key, base_url=args.api_base)

    json_file_path = './user_usage_3weeks_compressed.json'
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    json_str = json.dumps(data, ensure_ascii=False)

    user_prompt = prompt.USER_PATTERN_EXTRACTION_PROMPTS + "\n---3 weeks of user log---\n" + json_str
    messages = [
        {"role": "system", "content": prompt.SYSTEM_PATTERN_EXTRACTION_PROMPTS},
        {"role": "user", "content": user_prompt},
    ]

    stream = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream=True,
    )
    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        c0 = chunk.choices[0]
        delta = c0.delta
        if delta is None:
            continue
        piece = getattr(delta, "content", None) or ""
        if piece:
            parts.append(piece)
            print(piece, end="", flush=True)
    text = "".join(parts)
    print()

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
