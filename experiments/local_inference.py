import httpx
import time
import json


"""
Notes:
  - The generate endpoint requires a model. An empty/missing prompt produces no meaningful generation.
  - With Ollama's default streaming behavior, the response body contains multiple JSON objects/chunks, so 
    response.json() cannot parse the whole body as one JSON document.
  - .iter_lines() is used to iterate over the response body line by line, and each line is parsed as a separate JSON object.
"""


def main():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5-coder:14b",
        "prompt": "Hello",
        "stream": True,
    }
    with httpx.Client(timeout=None) as client:
        while True:
            prompt = input("Enter prompt (or 'exit' to quit): ")
            if prompt.lower() == "exit":
                break
            payload["prompt"] = prompt
            start = time.perf_counter()
            first_chunk_time = None
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if not data["done"]:
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - start
                        print(data["response"], end="", flush=True)
                    else:
                        total_wall_time = time.perf_counter() - start
                        eval_seconds = data["eval_duration"] / 1e9
                        tokens_per_second = data["eval_count"] / eval_seconds
                        print()
                        print(f"Input tokens: {data['prompt_eval_count']}")
                        print(f"Output tokens: {data['eval_count']}")
                        print(f"Output tokens/second: {tokens_per_second:.2f}")
                        if first_chunk_time is not None:
                            print(f"Time to first streamed chunk: {first_chunk_time:.2f} seconds")
                        print(f"Total wall time: {total_wall_time:.2f} seconds")
                        print(f"Response total latency: {data['total_duration']/1e9:.2f} seconds")
                        print(f"Response load latency: {data['load_duration']/1e9:.2f} seconds")
                        print(f"Response prompt eval latency: {data['prompt_eval_duration']/1e9:.2f} seconds")
                        print(f"Response eval latency: {data['eval_duration']/1e9:.2f} seconds")


if __name__ == "__main__":
    main()