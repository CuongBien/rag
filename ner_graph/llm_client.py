from llama_index.core.llms import LLMMetadata
from llama_index.llms.openai import OpenAI


class GroqOpenAI(OpenAI):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=128_000,
            num_output=self.max_tokens or 2048,
            is_chat_model=True,
            model_name=self.model,
        )


def create_llm(groq_model: str, groq_api_key: str, groq_api_base: str) -> OpenAI:
    print(f"[llm] Creating Groq OpenAI-compatible client, model={groq_model}")
    return GroqOpenAI(
        model=groq_model,
        temperature=0.0,
        max_retries=3,
        timeout=120.0,
        reuse_client=True,
        api_key=groq_api_key,
        api_base=groq_api_base,
    )
