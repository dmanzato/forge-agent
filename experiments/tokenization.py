import httpx
import json


"""
Observed:
  - Very short prompts still report ~30 input tokens through Ollama/Qwen, which suggests a fixed
    prompt/template overhead in addition to the user-provided text.
  - Because of that fixed overhead, absolute prompt token counts do not directly equal the token count
    of the sample string itself.
  - Relative differences are still meaningful: longer English text, code, and Portuguese produced
    higher prompt token counts than very short samples.
  - A follow-up experiment should subtract a fixed-prefix baseline so that the incremental token cost
    of each sample can be estimated more directly.
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
        "prompt": "Hello",
        "stream": True,
        "options": {
            "num_predict": 1,
            "num_ctx": 8192
        }
    }
    initialize_results_file()
    with httpx.Client(timeout=None) as client:
        for prompt, desc in zip(samples, [f"sample {i}" for i in range(len(samples))]):
            payload = payload_template.copy()
            make_stream_iteration(client, payload, desc, prompt, url)


def make_stream_iteration(client: httpx.Client, payload: dict[str, object], prompt_desc: str, prompt: str, url: str):
    print(f"Running : {prompt_desc}")
    payload["prompt"] = prompt
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