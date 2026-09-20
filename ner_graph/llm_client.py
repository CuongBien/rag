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

class OllamaOpenAI(OpenAI):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=32_768,
            num_output=self.max_tokens or 2048,
            is_chat_model=True,
            model_name=self.model,
        )

class GeminiOpenAI(OpenAI):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=1_000_000,
            num_output=self.max_tokens or 8192,
            is_chat_model=True,
            model_name=self.model,
        )

def create_llm(config=None, *args, **kwargs) -> object:
    import os

    # Hỗ trợ gọi kiểu legacy: create_llm(model, api_key, api_base)
    if isinstance(config, str):
        model = config
        api_key = args[0] if len(args) > 0 else kwargs.get("api_key", "ollama")
        api_base = args[1] if len(args) > 1 else kwargs.get("api_base", "http://localhost:11434/v1")
        return OllamaOpenAI(
            model=model,
            api_key=api_key or "ollama",
            api_base=api_base,
            temperature=0.0,
            timeout=300.0,
            max_retries=3,
        )

    llm_provider = getattr(config, "llm_provider", None) or os.getenv("LLM_PROVIDER", "ollama").lower()
    gemini_api_key = getattr(config, "gemini_api_key", None) or os.getenv("GEMINI_API_KEY", "")
    groq_api_key = getattr(config, "groq_api_key", None) or os.getenv("GROQ_API_KEY", "")

    # 1. Nếu chỉ định rõ Gemini hoặc có key Gemini (và không chọn ollama)
    if llm_provider == "gemini" or (gemini_api_key and llm_provider != "ollama"):
        gemini_model = getattr(config, "gemini_model", "gemini-2.5-flash-lite")
        model_name = gemini_model.replace("models/", "")
        print(f"[llm] Creating Gemini (OpenAI-compatible) client, model={model_name}")
        return GeminiOpenAI(
            model=model_name,
            api_key=gemini_api_key,
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0.0,
            max_retries=3,
        )

    # 2. Nếu chọn Groq và có API key
    if llm_provider == "groq" or (groq_api_key and llm_provider != "ollama"):
        groq_model = getattr(config, "groq_model", "llama-3.1-8b-instant")
        groq_api_base = getattr(config, "groq_api_base", "https://api.groq.com/openai/v1")
        print(f"[llm] Creating Groq OpenAI-compatible client, model={groq_model}")
        return GroqOpenAI(
            model=groq_model,
            temperature=0.0,
            max_retries=5,
            timeout=120.0,
            reuse_client=True,
            api_key=groq_api_key,
            api_base=groq_api_base,
        )

    # 3. Mặc định dùng Ollama chạy local (Qwen3 8B)
    ollama_model = getattr(config, "ollama_model", None) or os.getenv("OLLAMA_MODEL", "qwen3:8b")
    ollama_base_url = getattr(config, "ollama_base_url", None) or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    print(f"[llm] Creating Ollama client (model={ollama_model}, endpoint={ollama_base_url})")
    return OllamaOpenAI(
        model=ollama_model,
        api_key="ollama",
        api_base=ollama_base_url,
        temperature=0.0,
        timeout=300.0,
        max_retries=3,
    )
