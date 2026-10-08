import httpx
import time
import json
import copy


"""
Observed:
- A and C have similar fresh-prefill cost.
- B shares A's long prefix and has dramatically lower prefill cost.
- This demonstrates prefix/KV-cache reuse in the local inference runtime.
"""


RESULTS_FILENAME = "prompt_cache_results.txt"

PREFIX_A_SENTENCE = (
    "Distributed systems coordinate independent components over a network, "
    "requiring careful handling of latency, concurrency, partial failures, "
    "replication, consistency, retries, and fault tolerance. "
)

PREFIX_C_SENTENCE = (
    "Marine ecosystems contain interacting organisms across coral reefs, "
    "coastal habitats, open oceans, and deep-water environments, with energy "
    "flowing through complex food webs and nutrient cycles. "
)

REPETITIONS = 100

prefix_a = PREFIX_A_SENTENCE * REPETITIONS
prefix_c = PREFIX_C_SENTENCE * REPETITIONS

prompt_a = (
    prefix_a
    + "\n\nQuestion: What engineering challenges are mentioned above? "
      "Answer in one short sentence."
)

prompt_b = (
    prefix_a
    + "\n\nQuestion: What reliability concerns are mentioned above? "
      "Answer in one short sentence."
)

prompt_c = (
    prefix_c
    + "\n\nQuestion: What ecological concepts are mentioned above? "
      "Answer in one short sentence."
)

warmup = "Reply only with OK."

def initialize_results_file(filename=RESULTS_FILENAME):
    with open(filename, "w") as f:
        f.write("prompt  \tload_s\tttfc_s\tin_tks\tpf_s\tprefill_tps\t"
                "out_tks\tdec_s\tdec_tps\ttotal_s\n")


def record(data, filename=RESULTS_FILENAME):
    with open(filename, "a") as f:
        f.write(f"{data['prompt']}\t"
                f"{data['load_duration']:.2f}\t"
                f"{data['time_to_first_chunk']:.2f}\t"
                f"{data['input_tokens']}\t"
                f"{data['prefill_duration']:.2f}\t"
                f"{data['prefill_tps']:8.2f}\t"
                f"{data['output_tokens']}\t"
                f"{data['decode_duration']:.2f}\t"
                f"{data['decode_tps']:.2f}\t"
                f"{data['total_wall_time']:.2f}\n")


def main():
    url = "http://localhost:11434/api/generate"
    payload_template = {
        "model": "qwen2.5-coder:14b",
        "prompt": "Hello",
        "stream": True,
        "options": {
            "num_ctx": 8192
        }
    }
    initialize_results_file()
    with httpx.Client(timeout=None) as client:
        # Warm up the model without warming the test prefixes
        payload = copy.deepcopy(payload_template)
        payload["prompt"] = warmup
        make_stream_iteration(client, payload, "warmup  ", warmup, url)
        for prompt, desc in zip([prompt_a, prompt_b, prompt_c], ["prompt A", "prompt B", "prompt C"]):
            payload = copy.deepcopy(payload_template)
            make_stream_iteration(client, payload, desc, prompt, url)


def make_stream_iteration(client: httpx.Client, payload: dict[str, object], prompt_desc: str, prompt: str,
                          url: str):
    print(f"Running : {prompt_desc}")
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
                # print(data["response"], end="", flush=True)
            else:
                total_wall_time = time.perf_counter() - start
                load_seconds = data["load_duration"] / 1e9
                prefill_seconds = data["prompt_eval_duration"] / 1e9
                eval_seconds = data["eval_duration"] / 1e9
                prefill_tps = data["prompt_eval_count"] / prefill_seconds
                output_tps = data["eval_count"] / eval_seconds
                record({
                    "prompt": prompt_desc,
                    "load_duration": load_seconds,
                    "time_to_first_chunk": first_chunk_time,
                    "input_tokens": data["prompt_eval_count"],
                    "prefill_duration": prefill_seconds,
                    "prefill_tps": prefill_tps,
                    "output_tokens": data["eval_count"],
                    "decode_duration": eval_seconds,
                    "decode_tps": output_tps,
                    "total_wall_time": total_wall_time,
                })


if __name__ == "__main__":
    main()