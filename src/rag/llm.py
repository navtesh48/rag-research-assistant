import os
from anthropic import Anthropic

MODEL = os.getenv('CLAUDE_MODEL', 'claude-sonnet-5-5')
_client = None

def _c() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client

def structured(schema, system: str, user: str, max_tokens: int = 1500):
    """Force the model to return an instance of the pydantic model `schema` via a tool call."""
    tool = {'name': 'emit', 'description': 'Return the result.', 'input_schema': schema.model_json_schema()}
    r = _c().messages.create(model=MODEL, max_tokens=max_tokens, system=system, tools=[tool],
                             tool_choice={'type': 'tool', 'name': 'emit'},
                             messages=[{'role': 'user', 'content': user}])
    return schema.model_validate(next(b for b in r.content if b.type == 'tool_use').input)
