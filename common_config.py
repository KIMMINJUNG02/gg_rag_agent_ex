import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

api_key = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("BASE_URL")

def llm_connect(model : str):
    temperature = 0
    max_tokens = 512
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=temperature,
        use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        max_tokens=max_tokens,
    )