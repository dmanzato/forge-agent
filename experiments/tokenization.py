import httpx
import json


"""
Observed:
  - Ollama/Qwen adds fixed prompt/template overhead, so absolute prompt_eval_count values
    are not the raw token counts of the sample strings.
  - Using an identical fixed prefix allows the prefix/template overhead to be measured as a baseline.
  - Subtracting that baseline gives a practical estimate of each sample's incremental token cost.
  - Tokens are not equivalent to words: tokenization depends on the model tokenizer's vocabulary,
    subword segmentation, punctuation, language, and code syntax.
  - In this experiment, short strings such as "hello", Japanese text, and an emoji added only one token,
    while longer English, code, and Portuguese samples added more tokens.
"""

RESULTS_FILENAME = "tokenization_results.txt"

samples = [
    "hello",
    "hello world",
    "internationalization",
    "The quick brown fox jumps over the lazy dog.",
    "def calculate_total(items):",
    "Olá, tudo bem?",
    "こんにちは",
    "🚀",
]

prefix = "Count only the text after this marker:\n"


def initialize_results_file(filename=RESULTS_FILENAME):
    with open(filename, "w") as f:
        f.write("sample  \tin_tks\ttext\n")


def record(data, filename=RESULTS_FILENAME):
    with open(filename, "a") as f:
        f.write(f"{data['prompt']}\t"
                f"{data['input_tokens']}\t"
                f"{data['text']}\n")


def main():
    url = "http://localhost:11434/api/generate"
    payload_template = {
        "model": "qwen2.5-coder:14b",
        "prompt": prefix,
        "stream": True,
        "options": {
            "num_predict": 1,
            "num_ctx": 8192
        }
    }
    initialize_results_file()
    with httpx.Client(timeout=None) as client:
        payload = payload_template.copy()
        make_stream_iteration(client, payload, "baseline", "", url)
        for prompt, desc in zip(samples, [f"sample {i}" for i in range(len(samples))]):
            payload = payload_template.copy()
            make_stream_iteration(client, payload, desc, prompt, url)


def make_stream_iteration(client: httpx.Client, payload: dict[str, object], prompt_desc: str, prompt: str, url: str):
    print(f"Running : {prompt_desc}")
    payload["prompt"] += prompt
    with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data["done"]:
                record({
                    "prompt": prompt_desc,
                    "input_tokens": data["prompt_eval_count"],
                    "text": prompt,
                })


if __name__ == "__main__":
    main()