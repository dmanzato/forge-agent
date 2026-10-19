import httpx
import json
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from dataclasses import dataclass
from pathlib import Path


"""
Observed:

- Ollama structured output can be constrained with a Pydantic-generated JSON
  Schema so the model produces one of the supported typed tool-call shapes.

- The model's output remains untrusted even when it is schema-valid. For
  example, the model may preserve, normalize, or transform paths supplied in
  the user prompt.

- Pydantic validates the structure of the proposed action, while runtime policy
  independently determines whether that valid action is permitted.

- Tool paths are defined as workspace-relative. Absolute paths are denied even
  when they point inside the workspace.

- Relative paths containing ".." are not rejected syntactically. The runtime
  resolves the path and allows it only when the canonical target remains under
  the workspace root.

- Path resolution also follows symlinks. A relative path through a symlink that
  escapes the workspace is denied after canonicalization.

- File existence is an execution concern rather than a policy concern. A
  nonexistent path can be policy-allowed and subsequently fail with
  FILE_NOT_FOUND.

- write_file does not create missing parent directories.

- The policy-approved resolved path is passed to the executor instead of being
  independently resolved again during execution.

- Execution is guarded in two places: the orchestration layer only attempts an
  allowed call, and execute_tool_call() independently refuses a denied
  PolicyDecision.

- Tool execution returns a structured ToolResult that distinguishes successful
  execution from FILE_NOT_FOUND, PERMISSION_DENIED, IO_ERROR, policy denial,
  and other runtime failures.

- End-to-end flow demonstrated:
    natural-language request
      -> LLM structured ToolCall proposal
      -> Pydantic validation
      -> runtime policy evaluation
      -> conditional tool execution
      -> structured ToolResult

- Core runtime principle:
    The model proposes actions; the runtime validates, enforces, executes,
    and records them.

- The symlink-escape experiment assumes a test symlink was created manually
  beneath the workspace that points outside the workspace.
"""


RESULTS_FILENAME = "tool_calling_results.txt"

prompts = [
    "Read src/main.py",
    "Read README.md",
    "Write hello world to notes.txt",
    "Create tests/test_x.py containing assert True", # write_file does not create parent directories
    "Read ../../etc/passwd",
    "Read /etc/passwd",
    "Read ../../../tmp/foo",
    "Read src/../README.md",
    "Read ./symlink_that_escapes_root/notes.txt",
    "Read ./symlink_that_escapes_rook/notes.txt",
    "Read notes.txt"
]


@dataclass
class ToolResult:
    success: bool
    output: str | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class RuntimeContext:
    workspace_root: Path
    principal: str


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    resolved_path: Path | None = None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadFileArgs(StrictModel):
    path: str


class WriteFileArgs(StrictModel):
    path: str
    content: str


class ReadFileCall(StrictModel):
    tool: Literal["read_file"]
    arguments: ReadFileArgs


class WriteFileCall(StrictModel):
    tool: Literal["write_file"]
    arguments: WriteFileArgs


ToolCall = Annotated[
    Union[ReadFileCall, WriteFileCall],
    Field(discriminator="tool"),
]


tool_call_adapter = TypeAdapter(ToolCall)


def _write_file(resolved_path: Path, content: str) -> ToolResult:
    try:
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(success=True, output=None, error_code=None, error_message=None)
    except FileNotFoundError as e:
        return ToolResult(success=False, output=None, error_code="FILE_NOT_FOUND", error_message=str(e))
    except PermissionError as e:
        return ToolResult(success=False, output=None, error_code="PERMISSION_DENIED", error_message=str(e))
    except OSError as e:
        return ToolResult(success=False, output=None, error_code="IO_ERROR", error_message=str(e))


def _read_file(resolved_path: Path) -> ToolResult:
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolResult(success=True, output=content, error_code=None, error_message=None)
    except FileNotFoundError as e:
        return ToolResult(success=False, output=None, error_code="FILE_NOT_FOUND", error_message=str(e))
    except PermissionError as e:
        return ToolResult(success=False, output=None, error_code="PERMISSION_DENIED", error_message=str(e))
    except OSError as e:
        return ToolResult(success=False, output=None, error_code="IO_ERROR", error_message=str(e))


def execute_tool_call(
        tool_call: ToolCall,
        policy_decision: PolicyDecision,
) -> ToolResult:
    resolved_path = policy_decision.resolved_path

    if not policy_decision.allowed:
        return ToolResult(
            success=False,
            error_code="POLICY_DENIED",
            error_message=policy_decision.reason,
        )

    if resolved_path is None:
        return ToolResult(success=False, output=None, error_code="UNRESOLVED_PATH", error_message="The path could not be resolved")

    if tool_call.tool == "read_file":
        return _read_file(resolved_path)
    elif tool_call.tool == "write_file":
        return _write_file(resolved_path, tool_call.arguments.content)
    else:
        return ToolResult(success=False, output=None, error_code="UNKNOWN_TOOL", error_message=f"Unknown tool: {tool_call.tool}")


def evaluate_tool_call(
        tool_call: ToolCall,
        context: RuntimeContext,
) -> PolicyDecision:
    # Does the requested path resolve inside the configured workspace?
    requested_path = Path(tool_call.arguments.path)

    if requested_path.is_absolute():
        return PolicyDecision(
            allowed=False,
            reason="Absolute paths are not allowed",
            resolved_path=None
        )

    workspace_root = context.workspace_root.resolve()
    resolved_path = (workspace_root / requested_path).resolve()

    if not resolved_path.is_relative_to(workspace_root):
        return PolicyDecision(allowed=False, reason=f"Path {resolved_path} is outside the workspace", resolved_path=resolved_path)

    return PolicyDecision(allowed=True, reason="Path is inside the workspace", resolved_path=resolved_path)


def initialize_results_file(filename=RESULTS_FILENAME):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("")


def record(data, filename=RESULTS_FILENAME):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(
                f"-" * 80 + "\n"
                f"Run: {data['run']:<5}"
                f"Valid tool call: {data['valid_tool_call']:<5}"
                f"Prompt: {data['prompt']}\n"
                f"Raw response: {data['raw_response']}\n"
                f"Parsed tool call: {data['parsed_tool_call']}\n"
                f"Policy decision: {data['policy_decision']}\n"
                f"Tool attempted: {data['tool_attempted']}\n"
                f"Tool result: {data['tool_result']}\n"
        )


def main():
    initialize_results_file()
    schema = tool_call_adapter.json_schema()
    ctx = RuntimeContext(
        workspace_root=Path(__file__).resolve().parent,
        principal="local-user",
    )
    with httpx.Client(timeout=None) as client:
        for run, prompt in enumerate(prompts, start=1):
            make_stream_iteration(client, prompt, run, output_format=schema, ctx=ctx)


def make_stream_iteration(client: httpx.Client, prompt: str, run: int, output_format: object, ctx: RuntimeContext):
    print(f"-" * 80)
    print(f"Running : iteration {run}, prompt: {prompt}")
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
                valid_tool_call = False
                parsed_tool_call = None
                policy_decision = None
                tool_attempted = False
                tool_result = None
                try:
                    parsed_tool_call = tool_call_adapter.validate_json(answer)
                    valid_tool_call = True
                    policy_decision = evaluate_tool_call(parsed_tool_call, ctx)
                    print(f"Policy decision: {policy_decision}")

                    if policy_decision.allowed:
                        tool_result = execute_tool_call(parsed_tool_call, policy_decision)
                        tool_attempted = True
                        print(f"Tool result: {tool_result}")

                except ValidationError as e:
                    print(f"Validation failed for run {run}: {e}")
                record({
                    "run": run,
                    "prompt": prompt,
                    "raw_response": answer,
                    "valid_tool_call": valid_tool_call,
                    "parsed_tool_call": parsed_tool_call,
                    "policy_decision": policy_decision,
                    "tool_attempted": tool_attempted,
                    "tool_result": tool_result
                })


if __name__ == "__main__":
    main()