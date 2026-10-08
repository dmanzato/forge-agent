import httpx
import json


"""
Observed:
  - Separate inference requests do not automatically preserve conversation state, even when they use
    the same client, Ollama server, and loaded model.
  - To make prior conversation available to a later inference call, the application/agent runtime must
    explicitly include the relevant history in the new request.
  - Manually reconstructing the conversation through /api/generate allowed the model to recover the
    secret number.
  - Supplying the same history as structured role-based messages through /api/chat produced the same
    semantic result.
  - /api/chat still sends the conversation history as context; it provides a structured representation
    and lets the model's chat template handle role formatting.
  - Conversation state/context management belongs to the agent/application runtime, while the inference
    runtime owns tokenization, prefill, KV caching, and decoding.
"""


RESULTS_FILENAME = "context_management_results.txt"

requests = [
    "My secret number is 3719. Reply only with OK.",
    "What is my secret number? Reply only with the number.",
    "User: My secret number is 3719. Reply only with OK.\nAssistant: OK\nUser: What is my secret number? Reply only with the number.",
]

messages = [
    {
        "role": "user",
        "content": "My secret number is 3719. Reply only with OK.",
    },
    {
        "role": "assistant",
        "content": "OK",
    },
    {
        "role": "user",
        "content": "What is my secret number? Reply only with the number.",
    },
]

def initialize_results_file(filename=RESULTS_FILENAME):
    with open(filename, "w") as f:
        f.write("request  \tin_tks\ttext\n")


def record(data, filename=RESULTS_FILENAME):
    with open(filename, "a") as f:
        f.write(f"{data['prompt']}\t"
                f"{data['input_tokens']}\t"
                f"{data['text']}\n")
        f.write(f"response: {data['response']}\n")


def main():
    url_generate = "http://localhost:11434/api/generate"
    url_chat = "http://localhost:11434/api/chat"
    payload_template = {
        "model": "qwen2.5-coder:14b",
        "stream": True,
        "options": {
            "num_ctx": 8192
        }
    }
    initialize_results_file()
    with httpx.Client(timeout=None) as client:
        for i, request in enumerate(requests):
            payload = payload_template.copy()
            make_stream_iteration_through_generate(client, payload, f"request {i}", request, url_generate)
        payload = payload_template.copy()
        make_stream_iteration_through_chat(client, payload, "request 3", messages, url_chat)


def make_stream_iteration_through_generate(client: httpx.Client, payload: dict[str, object], prompt_desc: str,
                                           prompt: str, url: str):
    print(f"Running : {prompt_desc} through generate endpoint")
    payload["prompt"] = prompt
    with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        answer = ""
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if not data["done"]:
                answer += data["response"]
            else:
                record({
                    "prompt": prompt_desc,
                    "input_tokens": data["prompt_eval_count"],
                    "text": prompt,
                    "response": answer,
                })


def make_stream_iteration_through_chat(client: httpx.Client, payload: dict[str, object], prompt_desc: str,
                                       messages: list[dict[str, str]], url: str):
    print(f"Running : {prompt_desc} through chat endpoint")
    payload["messages"] = messages
    with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        answer = ""
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if not data["done"]:
                answer += data["message"]["content"]
            else:
                record({
                    "prompt": prompt_desc,
                    "input_tokens": data["prompt_eval_count"],
                    "text": messages,
                    "response": answer,
                })


if __name__ == "__main__":
    main()