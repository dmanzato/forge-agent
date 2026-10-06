import httpx
import time

"""
Notes:
  - The generate endpoint requires a model. An empty/missing prompt produces no meaningful generation.
  - With Ollama's default streaming behavior, the response body contains multiple JSON objects/chunks, so 
    response.json() cannot parse the whole body as one JSON document.
"""

def main():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5-coder:14b",
        "prompt": "Hello",
        "stream": False,
    }
    with httpx.Client(timeout=None) as client:
        prompt = None
        while prompt != "exit":
            prompt = input("Enter prompt (or 'exit' to quit): ")
            if prompt == "exit":
                break
            payload["prompt"] = prompt
            latency = time.perf_counter()
            response = client.post(url, json=payload)
            latency = time.perf_counter() - latency
            response.raise_for_status()
            data = response.json()
            print(data["response"])
            print(f"Input tokens: {data.get('prompt_eval_count')}")
            print(f"Output tokens: {data.get('eval_count')}")
            print(f"Measured latency: {latency:.2f} seconds; \n"
                  f"Response total latency: {data.get('total_duration')/10**9:.2f} seconds; \n"
                  f"Response load latency: {data.get('load_duration')/10**9:.2f} seconds; \n"
                  f"Response prompt eval latency: {data.get('prompt_eval_duration')/10**9:.2f} seconds; \n"
                  f"Response eval latency: {data.get('eval_duration')/10**9:.2f} seconds")



if __name__ == "__main__":
    main()