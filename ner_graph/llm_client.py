from llama_index.core.llms import LLMMetadata
from llama_index.llms.openai import OpenAI


from llama_index.llms.gemini import Gemini

class GroqOpenAI(OpenAI):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=128_000,
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

def create_llm(config) -> object:
    # Ưu tiên dùng Gemini nếu người dùng truyền GEMINI_API_KEY
    gemini_api_key = getattr(config, "gemini_api_key", None)
    import os
    gemini_api_key = os.environ.get("GEMINI_API_KEY", gemini_api_key)
    
    if gemini_api_key:
        print(f"[llm] Creating Gemini (OpenAI-compatible) client, model={config.gemini_model}")
        return GeminiOpenAI(
            model=config.gemini_model,
            api_key=gemini_api_key,
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0.0,
            max_retries=3,
        )
        
    print(f"[llm] Creating Groq OpenAI-compatible client, model={config.groq_model}")
    groq_api_key = config.groq_api_key
    groq_api_key = os.environ.get("GROQ_API_KEY", groq_api_key)
    
    return GroqOpenAI(
        model=config.groq_model,
        temperature=0.0,
        max_retries=5,
        timeout=120.0,
        reuse_client=True,
        api_key=groq_api_key,
        api_base=config.groq_api_base,
    )
