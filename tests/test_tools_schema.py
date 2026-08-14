from daemon.tools.schema import DELEGATE_TOOL_NAME, is_flat_schema


def test_flat_when_all_properties_are_primitive():
    assert is_flat_schema(
        {"type": "object", "properties": {"target": {"type": "string"}}}
    )
    assert is_flat_schema(
        {"type": "object", "properties": {"command": {"type": "string"},
                                          "cwd": {"type": "string"}}}
    )


def test_flat_allows_enum_and_number_and_bool():
    assert is_flat_schema(
        {"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["a", "b"]},
            "count": {"type": "integer"},
            "force": {"type": "boolean"}}}
    )


def test_nested_object_property_is_not_flat():
    assert not is_flat_schema(
        {"type": "object", "properties": {"parent": {"type": "object"}}}
    )


def test_array_property_is_not_flat():
    assert not is_flat_schema(
        {"type": "object", "properties": {"pages": {"type": "array"}}}
    )


def test_composed_schema_is_not_flat():
    assert not is_flat_schema(
        {"type": "object", "properties": {"x": {"type": "string"}},
         "anyOf": [{"required": ["x"]}]}
    )


def test_empty_or_missing_properties_is_flat():
    # A no-arg tool (get_time) is trivially flat and callable over voice.
    assert is_flat_schema({"type": "object", "properties": {}})
    assert is_flat_schema({"type": "object"})


def test_delegate_name_is_the_literal():
    assert DELEGATE_TOOL_NAME == "delegate_task"
