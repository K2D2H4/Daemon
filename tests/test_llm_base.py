from daemon.llm.base import ImageBlock, Message
from daemon.tools.base import ToolResult


def test_message_carries_images_and_defaults_empty():
    assert Message(role="user", content="hi").images == ()
    m = Message(role="user", content="look", images=(ImageBlock(b"\xff\xd8", "image/jpeg"),))
    assert m.images[0].media_type == "image/jpeg"


def test_tool_result_carries_images_and_defaults_empty():
    assert ToolResult(call_id="c", name="see_screen", content="ok").images == ()
    r = ToolResult(call_id="c", name="see_screen", content="ok", images=(ImageBlock(b"\xff\xd8"),))
    assert r.images[0].media_type == "image/jpeg"
