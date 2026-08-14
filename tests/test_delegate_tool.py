import pytest

from daemon.tools.delegate import DELEGATE_ACK, DelegateTask
from daemon.tools.schema import DELEGATE_TOOL_NAME


async def test_run_enqueues_the_request_and_returns_the_ack():
    seen = {}
    fired = []
    tool = DelegateTask(
        enqueue=lambda request: seen.setdefault("request", request) or 7,
        notify=lambda: fired.append(True),
    )
    out = await tool.run({"request": "노션에 하위 페이지 만들어줘"})
    assert seen["request"] == "노션에 하위 페이지 만들어줘"
    assert out == DELEGATE_ACK
    assert fired == [True]  # the worker was signalled


async def test_run_rejects_an_empty_request():
    from daemon.tools.base import ToolError

    tool = DelegateTask(enqueue=lambda request: 1)
    with pytest.raises(ToolError):
        await tool.run({"request": "   "})


def test_spec_is_named_and_flat():
    from daemon.tools.schema import is_flat_schema

    tool = DelegateTask(enqueue=lambda request: 1)
    assert tool.spec.name == DELEGATE_TOOL_NAME
    assert tool.risk == "safe"
    assert is_flat_schema(tool.spec.parameters)  # the voice model must be able to call it
