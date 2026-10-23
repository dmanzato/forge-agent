import experiments.tool_calling_eval as tool_calling_eval_module_name

from experiments.tool_calling_eval import (evaluate_tool_call, execute_tool_call, ReadFileCall, WriteFileCall,
                                           ReadFileArgs, WriteFileArgs, PolicyDecision, RuntimeContext, ToolResult,
                                           _read_file, _write_file)
from pathlib import Path
from unittest.mock import Mock


def get_valid_read_file_call() -> ReadFileCall:
    return ReadFileCall(
        tool="read_file",
        arguments=ReadFileArgs(path="sample.txt"),
    )


def get_valid_write_file_call() -> WriteFileCall:
    return WriteFileCall(
        tool="write_file",
        arguments=WriteFileArgs(path="sample.txt", content="Hello, world!"),
    )


def test_evaluate_tool_call_read_file(tmp_path):
    # Prepare
    tool_call = get_valid_read_file_call()
    context = RuntimeContext(workspace_root=tmp_path, principal="local_test_user")

    # Act
    policy_decision = evaluate_tool_call(tool_call, context)

    # Assert
    assert isinstance(policy_decision, PolicyDecision)
    assert policy_decision.allowed
    assert policy_decision.reason == "Path is inside the workspace"
    assert policy_decision.resolved_path == (tmp_path / "sample.txt").resolve()


def test_evaluate_tool_call_write_file(tmp_path):
    # Prepare
    tool_call = get_valid_write_file_call()
    context = RuntimeContext(workspace_root=tmp_path, principal="local_test_user")

    # Act
    policy_decision = evaluate_tool_call(tool_call, context)

    # Assert
    assert isinstance(policy_decision, PolicyDecision)
    assert policy_decision.allowed
    assert policy_decision.reason == "Path is inside the workspace"
    assert policy_decision.resolved_path == (tmp_path / "sample.txt").resolve()


def test_evaluate_tool_call_normalized_relative_path(tmp_path):
    # Prepare
    tool_call = ReadFileCall(
        tool="read_file",
        arguments=ReadFileArgs(path="src/../sample.txt"),
    )
    context = RuntimeContext(workspace_root=tmp_path, principal="local_test_user")

    # Act
    policy_decision = evaluate_tool_call(tool_call, context)

    # Assert
    expected_path = (tmp_path / "sample.txt").resolve()
    assert isinstance(policy_decision, PolicyDecision)
    assert policy_decision.allowed
    assert policy_decision.reason == "Path is inside the workspace"
    assert policy_decision.resolved_path == expected_path


def test_evaluate_tool_call_outside_workspace(tmp_path):
    # Prepare
    tool_call = ReadFileCall(
        tool="read_file",
        arguments=ReadFileArgs(path="../sample.txt"),
    )
    context = RuntimeContext(workspace_root=tmp_path, principal="local_test_user")

    # Act
    policy_decision = evaluate_tool_call(tool_call, context)

    # Assert
    expected_path = (tmp_path / "../sample.txt").resolve()
    assert isinstance(policy_decision, PolicyDecision)
    assert not policy_decision.allowed
    assert policy_decision.reason == f"Path {expected_path} is outside the workspace"
    assert policy_decision.resolved_path == expected_path


def test_evaluate_tool_call_absolute_path_not_allowed(tmp_path):
    # Prepare
    tool_call = ReadFileCall(
        tool="read_file",
        arguments=ReadFileArgs(path="/etc/passwd"),
    )
    context = RuntimeContext(workspace_root=tmp_path, principal="local_test_user")

    # Act
    policy_decision = evaluate_tool_call(tool_call, context)

    # Assert
    assert isinstance(policy_decision, PolicyDecision)
    assert not policy_decision.allowed
    assert policy_decision.reason == "Absolute paths are not allowed"
    assert not policy_decision.resolved_path


def test_evaluate_tool_call_symlink_outside_workspace(tmp_path):
    # Prepare
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"

    workspace.mkdir()
    outside.mkdir()

    # Create a symlink inside the workspace pointing outside
    link = workspace / "external_link"
    link.symlink_to(outside, target_is_directory=True)

    tool_call = ReadFileCall(
        tool="read_file",
        arguments=ReadFileArgs(path="external_link/sample.txt"),
    )
    context = RuntimeContext(workspace_root=workspace, principal="local_test_user")

    # Act
    policy_decision = evaluate_tool_call(tool_call, context)

    # Assert
    expected_resolved_path = (link / "sample.txt").resolve()
    assert isinstance(policy_decision, PolicyDecision)
    assert not policy_decision.allowed
    assert policy_decision.reason == f"Path {expected_resolved_path} is outside the workspace"
    assert policy_decision.resolved_path == expected_resolved_path


def test_execute_tool_call_denied_write(tmp_path):
    # Prepare
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"

    workspace.mkdir()
    outside.mkdir()

    # Create an external file containing "original content".
    external_file = outside / "sample.txt"
    external_file.write_text("original content")

    tool_call = WriteFileCall(
        tool="write_file",
        arguments=WriteFileArgs(path="../outside/sample.txt", content="replacement content"),
    )
    context = RuntimeContext(workspace_root=workspace, principal="local_test_user")

    # Act
    policy_decision = evaluate_tool_call(tool_call, context)
    tool_result = execute_tool_call(tool_call, policy_decision)

    # Assert
    expected_resolved_path = (outside / "sample.txt").resolve()
    assert isinstance(policy_decision, PolicyDecision)
    assert not policy_decision.allowed
    assert policy_decision.reason == f"Path {expected_resolved_path} is outside the workspace"
    assert policy_decision.resolved_path == expected_resolved_path

    # Verify that the external file's content remains unchanged
    assert not tool_result.success
    assert tool_result.error_code == "POLICY_DENIED"
    assert external_file.read_text() == "original content"


def get_valid_policy_decision(path: Path) -> PolicyDecision:
    return PolicyDecision(
        allowed=True,
        reason="Path is inside the workspace",
        resolved_path=(path / "sample.txt").resolve()
    )


def create_read_file_stub(tool_result: ToolResult, expected_path: Path):
    def _mock_read_file(resolved_path: Path) -> ToolResult:
        assert resolved_path == expected_path
        return tool_result

    return Mock(wraps=_mock_read_file)


def test_execute_tool_call_read_file(monkeypatch, tmp_path):
    # Prepare
    tool_call = get_valid_read_file_call()
    policy_decision = get_valid_policy_decision(tmp_path)
    stub = create_read_file_stub(
        ToolResult(success=True, output="Hello, world!", error_code=None, error_message=None),
        (tmp_path / "sample.txt").resolve()
    )
    monkeypatch.setattr(
        tool_calling_eval_module_name,
        "_read_file",
        stub
    )

    # Act
    tool_result = execute_tool_call(tool_call, policy_decision)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert tool_result.success
    assert tool_result.output == "Hello, world!"
    assert not tool_result.error_code
    assert not tool_result.error_message
    stub.assert_called_once_with((tmp_path / "sample.txt").resolve())


def test_execute_tool_call_read_file_file_not_found(monkeypatch, tmp_path):
    # Prepare
    tool_call = get_valid_read_file_call()
    policy_decision = get_valid_policy_decision(tmp_path)
    stub = create_read_file_stub(
        ToolResult(success=False, output=None, error_code="FILE_NOT_FOUND"),
        (tmp_path / "sample.txt").resolve()
    )
    monkeypatch.setattr(
        tool_calling_eval_module_name,
        "_read_file",
        stub
    )

    # Act
    tool_result = execute_tool_call(tool_call, policy_decision)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert not tool_result.success
    assert tool_result.output is None
    assert tool_result.error_code == "FILE_NOT_FOUND"
    stub.assert_called_once_with((tmp_path / "sample.txt").resolve())


def test_execute_tool_call_read_file_not_allowed(monkeypatch, tmp_path):
    # Prepare
    tool_call = get_valid_read_file_call()
    policy_decision = get_valid_policy_decision(tmp_path)
    stub = create_read_file_stub(
        ToolResult(success=False, output=None, error_code="PERMISSION_DENIED"),
        (tmp_path / "sample.txt").resolve()
    )
    monkeypatch.setattr(
        tool_calling_eval_module_name,
        "_read_file",
        stub
    )

    # Act
    tool_result = execute_tool_call(tool_call, policy_decision)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert not tool_result.success
    assert tool_result.output is None
    assert tool_result.error_code == "PERMISSION_DENIED"
    stub.assert_called_once_with((tmp_path / "sample.txt").resolve())


def test_execute_tool_call_read_file_io_error(monkeypatch, tmp_path):
    # Prepare
    tool_call = get_valid_read_file_call()
    policy_decision = get_valid_policy_decision(tmp_path)
    stub = create_read_file_stub(
            ToolResult(success=False, output=None, error_code="IO_ERROR"),
            (tmp_path / "sample.txt").resolve()
    )
    monkeypatch.setattr(
        tool_calling_eval_module_name,
        "_read_file",
        stub
    )

    # Act
    tool_result = execute_tool_call(tool_call, policy_decision)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert not tool_result.success
    assert tool_result.output is None
    assert tool_result.error_code == "IO_ERROR"
    stub.assert_called_once_with((tmp_path / "sample.txt").resolve())


def create_write_file_stub(tool_result: ToolResult, expected_path: Path, expected_content: str):
    def _mock_write_file(resolved_path: Path, content: str) -> ToolResult:
        assert resolved_path == expected_path
        assert content == expected_content
        return tool_result

    return Mock(wraps=_mock_write_file)


def test_execute_tool_call_write_file_file_not_found(monkeypatch, tmp_path):
    # Prepare
    tool_call = get_valid_write_file_call()
    policy_decision = get_valid_policy_decision(tmp_path)
    stub = create_write_file_stub(
        ToolResult(success=False, output=None, error_code="FILE_NOT_FOUND"),
        (tmp_path / "sample.txt").resolve(),
        "Hello, world!"
    )

    monkeypatch.setattr(
        tool_calling_eval_module_name,
        "_write_file",
        stub
    )

    # Act
    tool_result = execute_tool_call(tool_call, policy_decision)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert not tool_result.success
    assert tool_result.output is None
    assert tool_result.error_code == "FILE_NOT_FOUND"
    stub.assert_called_once_with((tmp_path / "sample.txt").resolve(), "Hello, world!")


def test_execute_tool_call_write_file_not_allowed(monkeypatch, tmp_path):
    # Prepare
    tool_call = get_valid_write_file_call()
    policy_decision = get_valid_policy_decision(tmp_path)
    stub = create_write_file_stub(
        ToolResult(success=False, output=None, error_code="PERMISSION_DENIED"),
        (tmp_path / "sample.txt").resolve(),
        "Hello, world!"
    )
    monkeypatch.setattr(
        tool_calling_eval_module_name,
        "_write_file",
        stub
    )

    # Act
    tool_result = execute_tool_call(tool_call, policy_decision)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert not tool_result.success
    assert tool_result.output is None
    assert tool_result.error_code == "PERMISSION_DENIED"
    stub.assert_called_once_with((tmp_path / "sample.txt").resolve(), "Hello, world!")


def test_execute_tool_call_write_file_io_error(monkeypatch, tmp_path):
    # Prepare
    tool_call = get_valid_write_file_call()
    policy_decision = get_valid_policy_decision(tmp_path)
    stub = create_write_file_stub(
        ToolResult(success=False, output=None, error_code="IO_ERROR"),
        (tmp_path / "sample.txt").resolve(),
        "Hello, world!"
    )
    monkeypatch.setattr(
        tool_calling_eval_module_name,
        "_write_file",
        stub
    )

    # Act
    tool_result = execute_tool_call(tool_call, policy_decision)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert not tool_result.success
    assert tool_result.output is None
    assert tool_result.error_code == "IO_ERROR"
    stub.assert_called_once_with((tmp_path / "sample.txt").resolve(), "Hello, world!")


def test_execute_tool_call_resolved_path_none():
    # Prepare
    tool_call = get_valid_read_file_call()
    policy_decision = PolicyDecision(
        allowed=True,
        reason="Path is inside the workspace",
        resolved_path=None
    )

    # Act
    tool_result = execute_tool_call(tool_call, policy_decision)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert not tool_result.success
    assert tool_result.output is None
    assert tool_result.error_code == "UNRESOLVED_PATH"


def test_execute_tool_call_not_allowed(tmp_path):
    # Prepare
    tool_call = get_valid_read_file_call()
    policy_decision = PolicyDecision(
        allowed=False,
        reason="Path is outside the workspace",
        resolved_path=(tmp_path / "sample.txt").resolve()
    )

    # Act
    tool_result = execute_tool_call(tool_call, policy_decision)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert not tool_result.success
    assert tool_result.output is None
    assert tool_result.error_code == "POLICY_DENIED"


def test_read_file_existing_file(tmp_path):
    # Prepare
    file_path = tmp_path / "sample.txt"
    file_content = "Hello, world!"
    file_path.write_text(file_content)

    # Act
    tool_result = _read_file(file_path)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert tool_result.success
    assert tool_result.output == file_content
    assert not tool_result.error_code
    assert not tool_result.error_message


def test_read_file_non_existing_file(tmp_path):
    # Prepare
    file_path = tmp_path / "non_existing.txt"

    # Act
    tool_result = _read_file(file_path)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert not tool_result.success
    assert tool_result.output is None
    assert tool_result.error_code == "FILE_NOT_FOUND"


def test_write_file_success(tmp_path):
    # Prepare
    file_path = tmp_path / "sample.txt"
    content = "Hello, world!"

    # Act
    tool_result = _write_file(file_path, content)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert tool_result.success
    assert tool_result.output is None
    assert not tool_result.error_code
    assert not tool_result.error_message

    # Verify that the file was written correctly
    assert file_path.read_text() == content


def test_write_file_missing_parent_directory(tmp_path):
    # Prepare
    file_path = tmp_path / "non_existing_dir" / "sample.txt"
    content = "Hello, world!"

    # Act
    tool_result = _write_file(file_path, content)

    # Assert
    assert isinstance(tool_result, ToolResult)
    assert not tool_result.success
    assert tool_result.output is None
    assert tool_result.error_code == "FILE_NOT_FOUND"

    # Verify that the file was not created
    assert not file_path.exists()
    
    