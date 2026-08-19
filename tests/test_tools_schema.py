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


def test_an_optional_primitive_is_still_flat():
    """`Optional[str]` renders as `anyOf: [string, null]`, and rejecting that shut
    voice out of most of the installed tool set for no reason.

    Measured 2026-08-19 against the live model, 20 trials per arm: offered
    `search_gmail_messages` with its real schema, the audio model emitted a correct
    call with both required arguments **20/20** - identically whether the `anyOf`
    was left as the server shipped it or folded to a plain type. The wall this gate
    exists for is genuine *nesting*; a nullable scalar is not it.
    """
    assert is_flat_schema(
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "page_token": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["query"],
        }
    )
    # `oneOf` is the same shape from a different generator, and JSON Schema's own
    # type-array spelling is a third.
    assert is_flat_schema(
        {"type": "object", "properties": {"x": {"oneOf": [{"type": "integer"}, {"type": "null"}]}}}
    )
    assert is_flat_schema({"type": "object", "properties": {"x": {"type": ["string", "null"]}}})


def test_a_union_that_hides_a_structure_is_still_not_flat():
    """The relaxation has to stay narrow. `send_gmail_message`'s `to` is
    `anyOf: [string, array, null]` - a caller may pass one address or many - and
    that is exactly the shape the model fakes rather than fills."""
    assert not is_flat_schema(
        {
            "type": "object",
            "properties": {
                "to": {"anyOf": [{"type": "string"}, {"type": "array"}, {"type": "null"}]}
            },
        }
    )
    assert not is_flat_schema(
        {"type": "object", "properties": {"x": {"anyOf": [{"type": "object"}, {"type": "null"}]}}}
    )
    # A bare null union carries no type to fill at all.
    assert not is_flat_schema(
        {"type": "object", "properties": {"x": {"anyOf": [{"type": "null"}]}}}
    )
    # `$ref` and `allOf` are real composition, not an optional scalar.
    assert not is_flat_schema(
        {"type": "object", "properties": {"x": {"allOf": [{"type": "string"}]}}}
    )
    assert not is_flat_schema({"type": "object", "properties": {"x": {"$ref": "#/defs/Thing"}}})
