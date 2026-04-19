from llama_index.core.llms import LLMMetadata
from llama_index.llms.openai import OpenAI


class GroqOpenAI(OpenAI):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=128_000,  # conservative window for LlamaIndex token accounting
            num_output=self.max_tokens or 2048,  # default max new tokens if unset
            is_chat_model=True,  # chat-completions API shape
            model_name=self.model,  # avoids OpenAI-only context-size lookup by id
        )


def create_llm(groq_model: str, groq_api_key: str, groq_api_base: str) -> OpenAI:
    # groq_model: provider model id; groq_api_key: secret; groq_api_base: OpenAI-compatible base URL.
    print(f"[llm] Creating Groq OpenAI-compatible client, model={groq_model}")
    return GroqOpenAI(
        model=groq_model,
        temperature=0.0,  # deterministic extraction/QA
        max_retries=3,  # transient HTTP failures
        timeout=120.0,  # seconds per request
        reuse_client=True,  # single HTTP client across calls
        api_key=groq_api_key,
        api_base=groq_api_base,
    )
