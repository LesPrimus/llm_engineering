from agent_from_scratch.models import LlmRequest, LlmResponse


class LlmClient:
    async def generate(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError
