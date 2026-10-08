import httpx
import time
import json


"""
Notes:
"""


RESULTS_FILENAME = "prompt_scaling_results.txt"
BASE_SENTENCE = (
    "Distributed systems coordinate independent components over a network, "
    "which introduces latency, partial failures, concurrency, and consistency tradeoffs. "
)
PROMPT_SIZES = {
    "tiny": 1,
    "small": 20,
    "medium": 100,
    "large": 500,
    "xlarge": 1000,
}


def make_prompt(repetitions: int) -> str:
    context = BASE_SENTENCE * repetitions
    return (
        context
        + "\n\nSummarize the text above in exactly one sentence."
    )


def initialize_results_file(filename=RESULTS_FILENAME):
    with open(filename, "w") as f:
        f.write("prompt\titer\tload_s\tttfc_s\tin_tks\tpf_s\tprefill_tps\t"
                "out_tks\tdec_s\tdec_tps\ttotal_s\n")


def record(data, filename=RESULTS_FILENAME):
    with open(filename, "a") as f:
        f.write(f"{data['prompt']}\t"
                f"{data['iteration']}\t"
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
        # Run a single iteration for the "tiny" prompt size to warm up the model and cache
        payload = payload_template.copy()
        payload["prompt"] = make_prompt(PROMPT_SIZES["tiny"])
        make_stream_iteration(client, -1, payload, "tiny", PROMPT_SIZES["tiny"], url, False)
        for prompt, size in PROMPT_SIZES.items():
            for i in range(5):  # Run 5 iterations for each prompt size
                payload = payload_template.copy()
                make_stream_iteration(client, i, payload, prompt, size, url)


def make_stream_iteration(client: httpx.Client, i: int, payload: dict[str, str | bool], prompt: str, size: int,
                          url: str, record_results: bool = True):
    print(f"Running prompt size: {prompt}, iteration: {i + 1}")
    payload["prompt"] = make_prompt(size)
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
                if record_results:
                    record({
                        "prompt": prompt,
                        "iteration": i + 1,
                        "load_duration": load_seconds,
                        "time_to_first_chunk": first_chunk_time,
                        "input_tokens": data['prompt_eval_count'],
                        "prefill_duration": prefill_seconds,
                        "prefill_tps": prefill_tps,
                        "output_tokens": data['eval_count'],
                        "decode_duration": eval_seconds,
                        "decode_tps": output_tps,
                        "total_wall_time": total_wall_time,
                    })


if __name__ == "__main__":
    main()