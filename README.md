# Pangea + Anthropic Python API library

A wrapper around the Anthropic Python library that wraps the Messages API with
Pangea AI Guard. Supports Python v3.10 and greater.

## Installation

```bash
pip install -U pangea-anthropic
```

## Usage

```python
import os
from pangea_anthropic import PangeaAnthropic

client = PangeaAnthropic(
    pangea_api_key=os.environ.get("PANGEA_API_KEY"),
    pangea_input_recipe="pangea_prompt_guard",
    pangea_output_recipe="pangea_llm_response_guard",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello, Claude",
        }
    ],
    model="claude-sonnet-4-20250514",
)
print(message.content)
```
