from __future__ import annotations

from typing import Generic
from unittest.mock import MagicMock

import pytest
from anthropic.types import Message, TextBlock, Usage
from pangea import PangeaResponseResult
from pangea.response import ResponseHeader, ResponseStatus
from pangea.services.ai_guard import TextGuardDetectors, TextGuardResult
from typing_extensions import TypeVar

from pangea_anthropic import PangeaAIGuardBlockedError, PangeaAnthropic

detectors = TextGuardDetectors()


T = TypeVar("T", bound=PangeaResponseResult)


class MockPangeaResponse(ResponseHeader, Generic[T]):
    """Lightweight mock of a PangeaResponse."""

    result: T | None = None

    def __init__(self, result: T, status: ResponseStatus = ResponseStatus.SUCCESS) -> None:
        super().__init__(status=status.value, request_id="", request_time="", response_time="", summary="")
        self.result = result


def test_pangea_anthropic_create_message_success() -> None:
    mock_ai_guard_client = MagicMock()
    mock_ai_guard_client.guard_text.return_value = MockPangeaResponse(
        result=TextGuardResult(blocked=False, transformed=False, prompt_messages=None, detectors=detectors)
    )

    mock_anthropic_client = MagicMock()
    mock_anthropic_client.messages.create.return_value = Message(
        id="msg_0123456789",
        content=[TextBlock(text="Hello, world!", type="text")],
        model="claude-sonnet-4-20250514",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    client = PangeaAnthropic(pangea_api_key="test_key")
    client.ai_guard_client = mock_ai_guard_client
    client.messages._post = mock_anthropic_client.messages.create

    response = client.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        model="claude-sonnet-4-20250514",
    )

    assert isinstance(response.content[0], TextBlock)
    assert response.content[0].text == "Hello, world!"
    assert mock_ai_guard_client.guard_text.call_count == 2


def test_pangea_anthropic_create_message_input_blocked() -> None:
    mock_ai_guard_client = MagicMock()
    mock_ai_guard_client.guard_text.return_value = MockPangeaResponse(
        result=TextGuardResult(blocked=True, transformed=False, prompt_messages=None, detectors=detectors)
    )

    client = PangeaAnthropic(pangea_api_key="test_key")
    client.ai_guard_client = mock_ai_guard_client

    with pytest.raises(PangeaAIGuardBlockedError):
        client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-sonnet-4-20250514",
        )

    mock_ai_guard_client.guard_text.assert_called_once()


def test_pangea_anthropic_create_message_output_blocked() -> None:
    mock_ai_guard_client = MagicMock()
    mock_ai_guard_client.guard_text.side_effect = [
        MockPangeaResponse(
            result=TextGuardResult(blocked=False, transformed=False, prompt_messages=None, detectors=detectors)
        ),
        MockPangeaResponse(
            result=TextGuardResult(blocked=True, transformed=False, prompt_messages=None, detectors=detectors)
        ),
    ]

    mock_anthropic_client = MagicMock()
    mock_anthropic_client.messages.create.return_value = Message(
        id="msg_0123456789",
        content=[TextBlock(text="I am a robot", type="text")],
        model="claude-sonnet-4-20250514",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    client = PangeaAnthropic(pangea_api_key="test_key")
    client.ai_guard_client = mock_ai_guard_client
    client.messages._post = mock_anthropic_client.messages.create

    with pytest.raises(PangeaAIGuardBlockedError):
        client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-sonnet-4-20250514",
        )

    assert mock_ai_guard_client.guard_text.call_count == 2
