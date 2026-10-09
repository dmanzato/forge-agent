import httpx
import json


"""
Observed:
  - With temperature=0.0, the model produced the same response across all runs in this experiment.
  - Increasing temperature introduced more variation in the generated responses.
  - At the same temperature (0.8), reducing top_p from 0.9 to 0.3 constrained the output space and
    produced a smaller set of repeated responses.
  - Setting a fixed seed made generation repeatable across runs for the same prompt and sampling
    configuration.
  - Different seeds produced different repeatable outputs in this experiment.
  - Temperature and top_p control sampling during decoding, while seed controls the pseudorandom sequence
    used by that sampling process.
"""


RESULTS_FILENAME = "sampling_results.txt"

prompt = "Write a one-sentence metaphor for debugging distributed systems."

configs_temp = [
    {"temperature": 0.0},
    {"temperature": 0.2},
    {"temperature": 0.8},
    {"temperature": 1.2},
]

configs_temp_top_p = [
    {"temperature": 0.8, "top_p": 0.9},
    {"temperature": 0.8, "top_p": 0.3}
]

configs_temp_seed = [
    {"temperature": 0.8, "seed": 42},
    {"temperature": 0.8, "seed": 123},
    {"temperature": 0.8, "seed": 713},
]


def initialize_results_file(filename=RESULTS_FILENAME):
    with open(filename, "w") as f:
        f.write("temp\ttop_p\tseed\trun\tresponse\n")


def record(data, filename=RESULTS_FILENAME):
    with open(filename, "a") as f:
        f.write(f"{data['temperature']}\t"
                f"{data.get('top_p', 'None')}\t"
                f"{data.get('seed', 'None')}\t"
                f"{data['run']}\t"
                f"{data['response']}\n")


def main():
    initialize_results_file()
    with httpx.Client(timeout=None) as client:
        for config in configs_temp:
            for run in range(5):  # Run 5 iterations for each configuration
                make_stream_iteration(client, prompt, config, run + 1)
        for config in configs_temp_top_p:
            for run in range(5):  # Run 5 iterations for each configuration
                make_stream_iteration(client, prompt, config, run + 1)
        for config in configs_temp_seed:
            for run in range(5):  # Run 5 iterations for each configuration
                make_stream_iteration(client, prompt, config, run + 1)


def make_stream_iteration(client: httpx.Client, prompt: str,
                          config: dict[str, float | int], run: int):
    temperature = config["temperature"]
    top_p = config.get("top_p", None)
    seed = config.get("seed", None)
    print(f"Running : temperature {temperature}, top_p {top_p}, seed {seed}, run {run}")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5-coder:14b",
        "stream": True,
        "prompt": prompt,
        "options": {
            "num_ctx": 8192,
            **config,
        }
    }
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
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "run": run,
                    "response": answer,
                })


if __name__ == "__main__":
    main()