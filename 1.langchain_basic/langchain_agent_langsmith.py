from langchain.agents import create_agent
import sys
sys.path.append("..") # 상위 폴더 path설정
from common_config import llm_connect

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = llm_connect(model="gpt-5.4-mini")

agent = create_agent(
    model=model,
    tools=[get_weather], # 함수 리스트
    system_prompt="You are a helpful assistant",
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
)
print(result)