import httpx
import json


"""
Observed:
  - Prompt instructions alone were not reliable for producing machine-parseable JSON; in this run,
    only 1 of 5 prompt-only responses was valid JSON with the expected fields.
  - Using Ollama's format="json" constrained generation to valid JSON; all 5 runs were parseable and
    contained the expected fields.
  - Supplying a JSON Schema through format also produced valid JSON with the expected fields in all
    5 runs.
  - json.loads(line) parses Ollama's streaming protocol envelope, while json.loads(answer) separately
    validates the model-generated content as JSON.
  - JSON mode constrains the output syntax to JSON, while schema mode additionally constrains the
    expected response structure.
  - The current expected-fields check verifies key presence only; stronger type/schema validation
    will be added later at the agent-runtime layer.
"""


RESULTS_FILENAME = "structured_output_results.txt"

prompt = """Analyze this code:

for item in items:
    print(item)

Return ONLY valid JSON with these fields:
- language
- complexity
- summary

Do not include Markdown or any other text."""

schema = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "complexity": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": [
        "language",
        "complexity",
        "summary",
    ],
}


def initialize_results_file(filename=RESULTS_FILENAME):
    with open(filename, "w") as f:
        f.write("mode\t\trun\tparseable\texpected_fields\n")


def record(data, filename=RESULTS_FILENAME):
    with open(filename, "a") as f:
        f.write(
                f"{data['mode']}\t"
                f"{data['run']}\t"
                f"{data['parseable']}\t\t"
                f"{data['expected_fields']}\n"
                )


def main():
    initialize_results_file()
    with httpx.Client(timeout=None) as client:
        for run in range(5):  # Run 5 iterations for each configuration
            make_stream_iteration(client, prompt, run + 1, output_format=None, mode="prompt_only")
        for run in range(5):  # Run 5 iterations for each configuration
            make_stream_iteration(client, prompt, run + 1, output_format="json", mode="json_mode")
        for run in range(5):  # Run 5 iterations for each configuration
            make_stream_iteration(client, prompt, run + 1, output_format=schema, mode="schema_mode")


def make_stream_iteration(client: httpx.Client, prompt: str, run: int, output_format: object, mode: str):
    print(f"Running : mode {mode}, iteration {run}")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5-coder:14b",
        "stream": True,
        "prompt": prompt,
    }
    if output_format is not None:
        payload["format"] = output_format
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
                parseable = False
                has_expected_fields = False
                try:
                    parsed_answer = json.loads(answer)
                    parseable = True
                    has_expected_fields = all(field in parsed_answer for field in ["language", "complexity", "summary"])
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON for run {run}: {answer}")
                record({
                    "mode": mode,
                    "run": run,
                    "response": answer,
                    "parseable": parseable,
                    "expected_fields": has_expected_fields
                })


if __name__ == "__main__":
    main()