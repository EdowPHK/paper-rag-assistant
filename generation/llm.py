from openai import OpenAI
import os

from config import get_config

def call_llm(prompt: str) -> str:
    cfg = get_config()

    base_url = str(cfg.get("llm_url", "https://api.deepseek.com"))
    model = str(cfg.get("llm_model", "deepseek-chat"))
    llm_api_key_env = str(cfg.get("llm_api_key_env", "DEEPSEEK_API_KEY"))

    api_key = os.getenv(llm_api_key_env, "")
    if not api_key:
        raise ValueError(f"{llm_api_key_env} must be set.")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt,}],
    )

    return response.output_text