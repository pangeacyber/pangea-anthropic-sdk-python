from __future__ import annotations

from typing import Any

from anthropic import Anthropic, AsyncAnthropic
from anthropic._compat import cached_property
from pangea import PangeaConfig
from pangea.asyncio.services import AIGuardAsync
from pangea.services import AIGuard
from typing_extensions import override

from pangea_anthropic.resources.messages import AsyncPangeaMessages, PangeaMessages

__all__ = ("PangeaAnthropic", "AsyncPangeaAnthropic")


class PangeaAnthropic(Anthropic):
    def __init__(
        self,
        *,
        pangea_api_key: str,
        pangea_input_recipe: str | None = None,
        pangea_output_recipe: str | None = None,
        pangea_base_url_template: str = "https://{SERVICE_NAME}.aws.us.pangea.cloud",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ai_guard_client = AIGuard(
            token=pangea_api_key, config=PangeaConfig(base_url_template=pangea_base_url_template)
        )
        self.pangea_input_recipe = pangea_input_recipe
        self.pangea_output_recipe = pangea_output_recipe

    @cached_property
    @override
    def messages(self) -> PangeaMessages:
        return PangeaMessages(self)


class AsyncPangeaAnthropic(AsyncAnthropic):
    def __init__(
        self,
        *,
        pangea_api_key: str,
        pangea_input_recipe: str | None = None,
        pangea_output_recipe: str | None = None,
        pangea_base_url_template: str = "https://{SERVICE_NAME}.aws.us.pangea.cloud",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ai_guard_client = AIGuardAsync(
            token=pangea_api_key, config=PangeaConfig(base_url_template=pangea_base_url_template)
        )
        self.pangea_input_recipe = pangea_input_recipe
        self.pangea_output_recipe = pangea_output_recipe

    @cached_property
    @override
    def messages(self) -> AsyncPangeaMessages:
        return AsyncPangeaMessages(self)
