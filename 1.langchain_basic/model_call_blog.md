# LangChain 모델 호출과 체인 구성: 메시지, 프롬프트, 출력 파서, Runnable

> 학습일: 2026년 10월 1일  
> 실습 자료: `01_mono_test.ipynb`~`05_runnable.ipynb`, `common_config.py`

LangChain으로 모델을 호출하는 것부터 시작해 메시지와 프롬프트를 구성하고, 응답을 원하는 데이터로 변환한 뒤 LCEL 체인으로 연결하는 방법을 정리했다. 각 예제에는 코드의 역할과 데이터가 바뀌는 과정을 설명했다.

이 글은 1~5번 노트북을 다룬다. 8번 노트북의 여섯 가지 프롬프트 기법과 비교 실험은 [프롬프트 엔지니어링 실습](./prompt_engineering_blog.md)으로 분리했다.

코드는 공통 설정을 실행한 뒤 같은 노트북 커널에서 순서대로 따라가는 구성이다. 원본을 보완한 코드와 추가 예제는 해당 위치에 표시했다. 출력 예시는 형태를 설명하기 위한 것이며, 실제 모델 응답은 달라질 수 있다.

## 1. LangChain을 왜 사용할까?

언어 모델을 이용하는 프로그램에는 모델 호출 외에도 여러 작업이 필요하다. 사용자 입력을 프롬프트에 넣고, 이전 대화를 함께 전달하고, 모델의 응답을 JSON이나 객체로 바꿔야 한다. 이후에는 응답을 가공하거나 여러 작업을 동시에 실행할 수도 있다.

LangChain은 이런 구성 요소를 연결해 LLM 애플리케이션을 만들 수 있도록 돕는 프레임워크다. 이번 실습의 중심은 다음 흐름이다.

```text
입력 데이터 → 프롬프트 생성 → 모델 호출 → 응답 파싱 → 필요한 후처리
```

예를 들어 사용자가 `{"topic": "LangChain"}`을 입력하면, 프롬프트가 이를 질문으로 바꾸고, 모델이 답변을 생성하고, 파서가 프로그램에서 사용하기 좋은 값으로 변환한다.

| 구성 요소 | 역할 | 이번 실습에서 사용하는 클래스 |
| --- | --- | --- |
| Chat Model | 메시지를 받아 응답 생성 | `ChatOpenAI` |
| Message | 시스템 지시, 사용자 발화, AI 응답 표현 | `SystemMessage`, `HumanMessage`, `AIMessage` |
| Prompt Template | 변수로 프롬프트 재사용 | `PromptTemplate`, `ChatPromptTemplate` |
| Output Parser | 모델 출력을 문자열·JSON·객체 등으로 변환 | `StrOutputParser`, `JsonOutputParser` 등 |
| Runnable | 입력을 받아 출력을 만드는 공통 실행 단위 | `RunnableLambda`, `RunnableParallel` 등 |
| LCEL | 실행 단위들을 연결하는 표현 방식 | `prompt \| llm \| parser` |

코드에서 패키지 이름도 구분해 두면 좋다. `langchain_core`에는 메시지, 프롬프트, 파서, Runnable 같은 핵심 인터페이스가 있고, `langchain_openai`에는 모델 연동을 위한 `ChatOpenAI`가 있다.

첫 번째 실습에서는 LangChain과 LangGraph의 차이를 질문했다. 개념적으로 LangChain은 모델과 애플리케이션 구성 요소를 연결하는 데 쓰이고, LangGraph는 상태를 유지하며 분기·반복하는 실행 흐름을 구성하는 데 쓰인다. 이번 글은 LangChain의 기초 구성 요소에 집중한다.

## 2. 실습 환경과 공통 모델 연결

참고한 노트북의 범위는 다음과 같다.

| 파일 | 학습 내용 |
| --- | --- |
| `01_mono_test.ipynb` | 공통 설정으로 모델 연결 확인 |
| `02_chat_model.ipynb` | `invoke`, 메시지, 스트리밍, 메타데이터 |
| `03_prompt_message.ipynb` | 프롬프트 템플릿, 대화 기록, 번역 |
| `04_output_parser.ipynb` | 문자열, JSON, Pydantic, 리스트 파서 |
| `05_runnable.ipynb` | LCEL, 입력 전달, 함수 연결, 병렬 실행 |

이 글은 프로젝트에서 확인한 Python 3.12 이상 환경과 `langchain 1.4.3`, `langchain-core 1.6.6`, `langchain-openai 1.6.7`, `pydantic 2.13.5`를 기준으로 정리했다. 이는 이 프로젝트의 설치 버전이며 최신 버전을 뜻하지 않는다.

### 2.1. 패키지 준비

저장소를 내려받아 따라 한다면 프로젝트 루트에서 잠금 파일에 맞춰 환경을 준비할 수 있다.

```bash
uv sync --locked
```

새로운 환경에서 시작한다면 필요한 패키지를 설치한다. 아래 명령은 버전을 고정하지 않으므로 저장소의 환경과 달라질 수 있다.

```bash
python -m pip install langchain langchain-openai python-dotenv ipykernel
```

노트북에서는 패키지가 설치된 Python 환경을 커널로 선택한다. 이 글의 Python 예제는 별도 표시가 없다면 공통 설정을 실행한 뒤 같은 커널에서 위에서 아래로 실행하는 방식이다.

### 2.2. 환경 변수 준비

프로젝트의 디렉터리 구조는 다음과 같다.

```text
프로젝트 루트/
├── .env
├── common_config.py
└── 1.langchain_basic/
    ├── 01_mono_test.ipynb
    ├── 02_chat_model.ipynb
    └── ...
```

루트의 `.env`에는 발급받은 연결 정보를 넣는다. 다음 값은 입력 위치를 보여 주기 위한 자리표시자다.

```dotenv
LLM_API_KEY=여기에_발급받은_API_키
BASE_URL=여기에_제공받은_API_기본_URL
```

이 프로젝트에서는 `LLM_API_KEY`를 코드에서 읽어 `api_key`에 직접 전달한다. 실제 키는 블로그나 저장소에 공개하지 않는다.

### 2.3. `common_config.py` 이해하기

프로젝트에 있는 공통 연결 코드는 다음과 같다. 주석과 줄 간격만 정리했다.

```python
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

api_key = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("BASE_URL")


def llm_connect(model: str):
    temperature = 0
    max_tokens = 512

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=temperature,
        use_responses_api=False,
        max_tokens=max_tokens,
    )
```

`load_dotenv()`는 `.env`의 값을 환경 변수로 읽는다. `override=True`이면 같은 이름의 기존 환경 변수도 파일의 값으로 덮어쓴다. `os.getenv()`는 해당 값을 꺼내며, 값이 없다면 `None`을 반환한다.

`llm_connect()`는 모델 이름을 받아 `ChatOpenAI` 객체를 반환한다. 이 객체를 만드는 것과 실제로 모델을 호출하는 것은 별개다. 이후 `invoke()`나 `stream()` 등을 실행할 때 요청이 발생한다.

| 설정 | 의미와 이 실습에서의 사용 |
| --- | --- |
| `model` | 라우터에서 사용할 모델 식별자 |
| `api_key` | 요청 인증에 사용할 키 |
| `base_url` | 요청을 보낼 API 기본 주소 |
| `temperature` | 지원 모델에서 출력의 무작위성을 조절하는 값 |
| `max_tokens` | 응답 생성에 사용할 토큰 예산을 제한하는 설정 |
| `use_responses_api=False` | 이 프로젝트에서 Chat Completions 방식을 사용하도록 지정 |

`temperature=0`은 비교적 일관된 응답을 얻기 위한 설정이지, 모든 실행에서 동일한 결과를 보장하는 설정은 아니다. 지원하는 매개변수와 토큰 한도의 해석은 모델 및 제공자에 따라 달라질 수 있다. 표준 호출 방식과 주요 설정은 [LangChain Models 문서](https://docs.langchain.com/oss/python/langchain/models)에서도 확인할 수 있다.

원본의 `use_responses_api=False`는 MonoRouter 연결 설정을 따른 것이다. `base_url`을 사용하는 모든 서비스가 반드시 이 값을 요구한다는 뜻은 아니다. 모델 이름 `gpt-5.4-mini` 역시 수업 코드의 값을 유지했으며, 실제 이용 가능 여부는 사용하는 라우터 설정에 따른다.

### 2.4. 01번 실습: 연결 확인

```python
import sys

sys.path.append("..")

from common_config import llm_connect

model = "gpt-5.4-mini"
llm = llm_connect(model=model)

response = llm.invoke("LangGraph와 LangChain의 차이를 설명해줘")
print(response.content)
```

`sys.path.append("..")`는 Python의 모듈 탐색 경로에 상위 폴더를 추가한다. 노트북의 현재 작업 디렉터리가 `1.langchain_basic`일 때 상위 폴더의 `common_config.py`를 불러올 수 있다. 작업 디렉터리가 프로젝트 루트라면 이 경로 추가는 필요 없다.

`llm.invoke()`는 질문을 보내고 완성된 응답을 받는다. 반환값은 일반적으로 `AIMessage`이며, 이번 텍스트 실습에서는 `response.content`로 답변 본문을 출력한다.

이 단계에서 확인할 것은 API 키와 주소를 읽었는지, 모델 식별자가 맞는지, 실제 응답이 돌아오는지다.

### 2.5. 원본에서 주의할 설정: 변수 선언과 인자 전달은 다르다

02번 노트북에는 다음 코드가 있다.

```python
model = "gpt-5.4-mini"
temperature = 0
max_tokens = 2000

llm = llm_connect(model=model)
print(llm)
```

여기서 `max_tokens = 2000`을 선언해도 연결 함수에 전달하지 않았으므로 실제 설정은 바뀌지 않는다. 현재 `llm_connect()`는 `model`만 인자로 받으며, 함수 내부에서 `max_tokens=512`를 사용한다.

아래는 긴 응답이 필요한 실험을 위한 **추가 예제**다. 원본 파일을 수정하지 않고 별도의 함수로 설정값을 전달하는 방법을 보여 준다.

```python
import os
from langchain_openai import ChatOpenAI


def llm_connect_custom(
    model: str,
    temperature: float = 0,
    max_tokens: int = 2000,
):
    return ChatOpenAI(
        model=model,
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["BASE_URL"],
        temperature=temperature,
        max_tokens=max_tokens,
        use_responses_api=False,
    )
```

이 함수는 `.env`를 읽은 상태에서 사용한다. `os.environ[...]`은 필수 환경 변수가 없으면 바로 오류를 내므로 누락을 확인하기 쉽다. 긴 응답이 필요한 경우 `llm = llm_connect_custom(model="gpt-5.4-mini", max_tokens=2000)`처럼 호출할 수 있다.

## 3. Chat Model 사용하기 — 02번 노트북

### 3.1. 문자열로 질문하기

```python
response = llm.invoke("LangChain이 무엇인지 한 문장으로 설명해줘")

print(response)
print(response.content)
```

문자열을 전달하면 모델이 사용자 메시지로 처리한다. `print(response)`는 메시지 객체 전체를 보여 주므로 본문 외의 정보도 확인할 수 있다. `print(response.content)`는 본문을 확인할 때 사용한다.

일반적인 텍스트 응답에서는 `content`가 문자열이지만, 멀티모달 응답 등에서는 콘텐츠 블록 목록일 수도 있다. 이번 예제는 텍스트 응답을 전제로 한다.

### 3.2. 역할을 구분해 메시지 보내기

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="당신은 친절한 파이썬 튜터입니다."),
    HumanMessage(content="리스트 컴프리헨션을 초보자에게 설명해줘"),
]

response = llm.invoke(messages)
print(response.content)
```

`SystemMessage`에는 AI의 역할이나 답변 지침을 넣고, `HumanMessage`에는 사용자의 질문을 넣는다. 이렇게 나누면 반복해서 사용할 지시와 매번 바뀌는 질문을 분리할 수 있다.

이전 모델 응답을 대화 기록에 담을 때는 `AIMessage`를 사용한다. 역할 정보는 단순히 문자열 앞에 붙이는 이름표가 아니라 모델에 전달하는 메시지 구조의 일부다.

### 3.3. 노트북에서 Markdown으로 보기

```python
from IPython.display import Markdown, display

display(Markdown(response.content))
```

`Markdown()`은 답변을 Markdown 표시 객체로 만들고, `display()`는 이를 노트북에 렌더링한다. 응답 안에 제목이나 목록이 있다면 읽기 편한 형태로 표시된다. 이 코드는 노트북 화면에 보여 주는 코드이며 `.md` 파일을 저장하는 코드는 아니다.

### 3.4. `stream()`으로 응답을 조금씩 받기

원본은 최신 LangChain 버전을 질문하지만, 검색 도구를 연결하지 않은 모델 답변만으로 최신 정보를 확인할 수는 없다. 여기서는 스트리밍 동작에 집중하도록 질문을 바꿨다.

```python
for chunk in llm.stream("LangChain의 주요 구성 요소를 설명해줘"):
    print(chunk.content, end="", flush=True)
print()
```

`stream()`은 응답이 생성되는 동안 `AIMessageChunk`를 차례로 반환한다. 청크는 응답의 일부이며, 한 청크가 반드시 한 토큰과 일치하는 것은 아니다.

`end=""`는 청크마다 줄바꿈하지 않고 이어 출력하도록 한다. `flush=True`는 Python 출력 버퍼를 즉시 비워 화면에 빨리 보이도록 한다. 모델의 생성 속도를 바꾸는 설정은 아니다.

### 3.5. 응답과 사용량 메타데이터 확인

```python
response = llm.invoke("안녕")

print("content:", response.content)
print("usage:", response.usage_metadata)
```

`usage_metadata`에는 제공자가 전달한 입력·출력 토큰 사용량 등이 들어갈 수 있다. 다만 라우터나 모델이 사용량을 전달하지 않으면 값이 없을 수 있다. 토큰 수는 글자 수나 단어 수와 같지 않으며, 요금 자체를 뜻하지도 않는다.

## 4. 프롬프트와 메시지 재사용하기 — 03번 노트북

프롬프트를 매번 문자열로 직접 작성하면 비슷한 코드가 반복된다. 프롬프트 템플릿을 사용하면 고정된 문장과 바뀌는 값을 분리할 수 있다.

03번 노트북은 `.env`를 다음처럼 직접 읽는 코드도 포함한다.

```python
import os
from dotenv import load_dotenv

load_dotenv(override=True, dotenv_path="../.env")
api_key = os.getenv("LLM_API_KEY")
```

현재 작업 디렉터리가 `1.langchain_basic`일 때의 상대 경로다. 이번 글의 공통 설정은 이미 환경 변수를 읽으므로 다시 실행할 필요는 없다. 또한 이 변수는 선언만으로 모델에 전달되지 않는다. 실제 연결은 `common_config.py`에서 읽은 값을 사용한다.

### 4.1. `PromptTemplate`: 문자열 템플릿

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "{topic}에 대해 한 문장으로 설명해줘"
)

prompt_value = template.invoke({"topic": "RAG"})

print(prompt_value)
print(prompt_value.to_string())
```

`{topic}`은 실행할 때 채울 변수다. `invoke({"topic": "RAG"})`를 호출하면 변수에 값이 들어간다. 반환값은 단순 문자열이 아니라 `StringPromptValue`이며, `.to_string()`으로 문자열 표현을 확인할 수 있다.

이 시점에는 모델을 호출하지 않았다. 프롬프트 템플릿의 `invoke()`는 프롬프트를 만드는 작업이다. 같은 메서드 이름이어도 대상 객체에 따라 하는 일이 다르다.

### 4.2. `ChatPromptTemplate`: 역할 기반 템플릿

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다."),
    ("human", "{question}"),
])

messages = chat_prompt.invoke({
    "role": "파이썬",
    "question": "데코레이터가 뭔가요?",
})

print(messages)
print(messages.to_messages())
```

`from_messages()`는 역할과 내용으로 구성된 템플릿을 만든다. 위 코드에는 `role`, `question` 두 변수가 있으므로 실행할 때 두 값을 모두 전달한다.

`messages`라는 변수명을 사용했지만 반환 객체는 `ChatPromptValue`다. `.to_messages()`를 호출하면 실제 메시지 목록을 확인할 수 있다.

### 4.3. 완성된 프롬프트를 모델에 전달하기

```python
from common_config import llm_connect

llm = llm_connect(model="gpt-5.4-mini")

messages = chat_prompt.invoke({
    "role": "머신러닝",
    "question": "과적합이 뭔가요?",
})

response = llm.invoke(messages)
print(response.content)
```

이제 같은 템플릿을 머신러닝 질문에 재사용했다. 먼저 템플릿이 메시지를 만들고, 그 결과를 모델이 받는다. `ChatPromptValue`는 모델에 직접 전달할 수 있으므로 꼭 메시지 목록으로 변환할 필요는 없다.

### 4.4. `MessagesPlaceholder`: 대화 기록 넣기

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

history_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

history = [
    HumanMessage(content="내 이름은 홍길동이야"),
    AIMessage(content="안녕하세요, 홍길동님!"),
]

question = "내 이름이 뭐라고?"
messages = history_prompt.invoke({
    "history": history,
    "question": question,
})

response = llm.invoke(messages)
print(response.content)
```

`MessagesPlaceholder`는 지정된 위치에 여러 메시지를 삽입한다. 위 예제에서는 시스템 메시지 다음에 이전 대화가 들어가고, 마지막에 새 질문이 들어간다.

모델이 이름을 답할 수 있는 이유는 과거 메시지를 이번 요청에 함께 보냈기 때문이다. 이 객체가 대화를 자동으로 저장하거나 기억해 주는 것은 아니다.

```python
# 노트북에서는 변수 이름만 적어도 객체를 확인할 수 있다.
response
```

```python
# 이번 대화를 다음 호출에서 사용할 기록에 추가한다.
history.append(HumanMessage(content=question))
history.append(AIMessage(content=response.content))

print(history)
```

원본처럼 응답 본문으로 새 `AIMessage`를 만들 수 있다. 메시지의 메타데이터까지 보존하려면 `history.append(response)`로 응답 객체를 그대로 추가할 수도 있다. 수정한 기록은 다음에 프롬프트를 만들 때 다시 전달해야 한다.

### 4.5. 여러 변수를 사용하는 번역 템플릿

```python
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {language} 번역가입니다."),
    ("human", "다음 문장을 번역해줘: {sentence}"),
])

messages = translation_prompt.invoke({
    "language": "영어",
    "sentence": "오늘 날씨가 정말 좋네요.",
})

response = llm.invoke(messages)
print(response.content)
```

`language`는 번역 언어, `sentence`는 원문이다. 변수만 바꾸면 같은 템플릿으로 다른 문장을 번역할 수 있다.

원본의 추가 문제는 출발 언어와 도착 언어까지 구분한다. 아래는 원본 변수명 `sorted_language`를 의미가 분명한 `source_language`로 바꾸고, 원문과 번역문을 모두 출력하도록 지시를 정리한 버전이다.

```python
translation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 {source_language}에서 {target_language}로 번역하는 번역가입니다.\n"
        "출력에는 원문({source_language})과 번역문({target_language})을 "
        "각각 표시하세요.",
    ),
    ("human", "다음 문장을 번역해줘: {text}"),
])

messages = translation_prompt.invoke({
    "source_language": "한국어",
    "target_language": "영어",
    "text": "오늘 날씨가 정말 좋네요.",
})
response = llm.invoke(messages)
print(response.content)

messages = translation_prompt.invoke({
    "source_language": "한국어",
    "target_language": "일본어",
    "text": "난 너무너무 졸려요",
})
response = llm.invoke(messages)
print(response.content)
```

템플릿의 변수 이름과 `invoke()`에 전달하는 딕셔너리 키는 반드시 일치해야 한다. 영어 번역 예시를 고정해 두는 대신 출력 규칙을 언어 변수로 표현해 일본어 번역에도 그대로 재사용했다.

## 5. 모델의 출력을 데이터로 바꾸기 — 04번 노트북

모델의 답변을 화면에 보여 주는 것과 프로그램에서 활용하는 것은 다르다. 영화 제목과 점수를 각각 저장하려면 답변을 일정한 구조로 읽어야 한다. 이때 Output Parser를 사용한다.

| 파서 | 이번 예제의 결과 | 사용 목적 |
| --- | --- | --- |
| `StrOutputParser` | `str` | 텍스트 응답 사용 |
| `JsonOutputParser` | `dict` | JSON 객체를 Python 데이터로 변환 |
| `PydanticOutputParser` | Pydantic 모델 인스턴스 | 필드 구조와 제약 검증 |
| `CommaSeparatedListOutputParser` | `list[str]` | 쉼표로 구분한 항목을 목록으로 변환 |

### 5.1. `StrOutputParser`: 문자열 결과 얻기

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("human", "{topic}을 한 문장으로 설명해줘"),
])
parser = StrOutputParser()

messages = prompt.invoke({"topic": "LangChain"})
response = llm.invoke(messages)
result = parser.invoke(response)

print(type(result))  # <class 'str'>
print(result)
```

각 줄을 따라가면 입력 딕셔너리가 프롬프트로, 프롬프트가 모델 응답으로, 모델 응답이 문자열로 바뀐다. 이번처럼 일반 텍스트 응답에서는 `.content`를 꺼내는 것과 비슷하게 보이지만, 파서 자체를 체인의 한 단계로 연결할 수 있다는 장점이 있다.

이 파서는 답변을 요약하거나 내용의 정확성을 검사하지 않는다.

### 5.2. `JsonOutputParser`: JSON 읽기

```python
from langchain_core.output_parsers import JsonOutputParser

prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        "파이썬 언어에 대한 정보를 name, year, creator 키를 가진 "
        "JSON으로 반환해줘. JSON만 반환해.",
    ),
])
parser = JsonOutputParser()

messages = prompt.invoke({})
response = llm.invoke(messages)
result = parser.invoke(response)

print(type(result))  # JSON 객체를 반환했다면 <class 'dict'>
print(result)
```

이 프롬프트에는 변수가 없지만 `invoke()`의 입력 인자는 필요하므로 빈 딕셔너리를 전달한다. 모델에는 JSON 출력을 요청하고, 파서는 응답 텍스트를 Python 데이터로 변환한다.

다음은 결과 형태를 보여 주는 예시다.

```python
{"name": "Python", "year": 1991, "creator": "Guido van Rossum"}
```

`JsonOutputParser`가 항상 딕셔너리를 반환하는 것은 아니다. JSON 배열이면 리스트가 된다. 또한 위 코드의 파서는 `name`, `year`, `creator`가 반드시 존재하는지나 각 필드의 타입까지 강제하지 않는다. 그런 검증에는 스키마가 필요하다.

### 5.3. `PydanticOutputParser`: 원하는 구조 정의하기

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class MovieReview(BaseModel):
    title: str = Field(description="영화 제목")
    score: int = Field(ge=0, le=10, description="0~10점 사이의 정수 점수")
    summary: str = Field(description="한 줄 감상")


parser = PydanticOutputParser(pydantic_object=MovieReview)
print(parser.get_format_instructions())
```

`BaseModel`을 상속한 클래스가 원하는 출력 구조다. `title`과 `summary`는 문자열, `score`는 정수 필드로 정의했다. `Field(description=...)`는 필드를 설명하며, 파서가 만드는 형식 지시사항에도 활용된다.

원본은 점수 설명만 있었지만, 여기서는 `ge=0`, `le=10`을 추가했다. 설명에 “10점 만점”이라고 쓰는 것만으로는 범위가 검증되지 않기 때문이다. 숫자 범위를 실제 제약으로 표현해야 범위를 벗어난 값이 검증에 실패한다.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 영화 평론가입니다.\n{format_instructions}"),
    ("human", "{movie} 영화를 리뷰해줘"),
])

messages = prompt.invoke({
    "movie": "인터스텔라",
    "format_instructions": parser.get_format_instructions(),
})

response = llm.invoke(messages)
result = parser.invoke(response)

print(type(result))
print(result.title)
print(result.score)
print(result.summary)
```

`get_format_instructions()`는 모델에게 보여 줄 출력 형식 설명을 만든다. 파서가 이 설명을 자동으로 모델에 보내는 것은 아니므로 프롬프트에 직접 넣는다.

파싱과 검증이 성공하면 `result`는 `MovieReview` 객체가 된다. 딕셔너리의 `result["title"]` 대신 `result.title`처럼 접근한다. Pydantic은 설정에 따라 타입을 변환할 수도 있으므로 항상 엄격한 원본 타입 검사와 같지는 않다. 또한 구조가 맞는 것과 리뷰 내용이 사실인 것은 별개의 문제다.

### 5.4. `CommaSeparatedListOutputParser`: 항목 목록 만들기

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        "파이썬 웹 프레임워크 5개를 쉼표로 구분해서 나열해줘. "
        "설명 없이 이름만.",
    ),
])

messages = prompt.invoke({})
response = llm.invoke(messages)
result = parser.invoke(response)

print(type(result))  # <class 'list'>
print(result)
```

모델이 `Django, Flask, FastAPI, ...`처럼 응답하면 문자열 목록으로 나눌 수 있다. 원본의 “Top 5”는 순위 기준이 없어 여기서는 단순히 다섯 개를 요청하도록 바꿨다.

이 파서는 목록으로 분리하는 역할을 한다. 항목이 정확히 다섯 개인지, 모두 실제 프레임워크인지까지 검증하지는 않는다.

### 5.5. 파싱 실패를 처리하는 방법

다음은 **추가 예제**다. 모델이 지시를 지키지 않거나 출력이 중간에 잘리면 파싱에 실패할 수 있으므로 오류를 처리한다.

```python
from langchain_core.exceptions import OutputParserException

json_parser = JsonOutputParser()

try:
    parsed = json_parser.invoke("JSON 대신 일반 문장으로 답했습니다.")
except OutputParserException as error:
    print("출력 형식을 확인해야 합니다:", error)
```

파서는 잘못된 모든 출력을 자동으로 고쳐 주지 않는다. 실제 서비스에서는 재시도하거나 사용자에게 오류를 전달하는 흐름을 설계해야 한다.

## 6. Runnable과 LCEL로 연결하기 — 05번 노트북

지금까지는 프롬프트, 모델, 파서를 각각 호출했다. Runnable은 이런 구성 요소를 같은 실행 방식으로 다룰 수 있게 하는 인터페이스다. LCEL은 Runnable을 `|` 연산자 등으로 조합하는 표현 방식이다.

### 6.1. 기본 체인: `prompt | llm | parser`

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("human", "{topic}에 대해 한 문장으로 설명해줘"),
])
parser = StrOutputParser()

basic_chain = prompt | llm | parser

result = basic_chain.invoke({"topic": "벡터 데이터베이스"})
print(result)
```

`|`는 앞 단계의 출력을 다음 단계의 입력으로 전달한다. 이 코드에서는 다음과 같이 데이터 형태가 바뀐다.

```text
{"topic": "벡터 데이터베이스"}
    → ChatPromptValue
    → AIMessage
    → str
```

체인을 정의하는 줄에서는 아직 모델을 호출하지 않는다. `basic_chain.invoke(...)`가 전체 흐름을 실행한다. 단계끼리 연결하려면 앞 단계의 출력 형태를 다음 단계가 받을 수 있어야 한다.

질문에 등장하는 벡터는 이 문맥에서 여러 숫자를 순서대로 담은 표현이다. 임베딩은 텍스트 등을 이런 수치 벡터로 표현하는 과정 또는 결과이고, 벡터 데이터베이스는 벡터를 저장하고 유사도를 이용해 검색하는 데 활용된다. 이 실습은 해당 개념을 모델에 질문하는 예제이며, 실제 임베딩 생성이나 검색을 구현한 것은 아니다.

### 6.2. 체인도 스트리밍할 수 있다

```python
for chunk in basic_chain.stream({"topic": "임베딩"}):
    print(chunk, end="", flush=True)
print()
```

모델만 스트리밍할 때는 `chunk.content`를 출력했다. 지금은 뒤에 `StrOutputParser()`가 연결되어 있으므로 `chunk` 자체가 문자열이다.

원본은 `flush=False`도 비교한다.

```python
for chunk in basic_chain.stream({"topic": "임베딩"}):
    print(chunk, end="", flush=False)
print()
```

`False`에서는 출력 시점을 버퍼링에 맡긴다. 화면에서 얼마나 차이가 보이는지는 노트북이나 터미널 환경에 따라 다르다. 두 반복문을 모두 실행하면 모델 요청도 두 번 발생하므로 출력 속도뿐 아니라 생성 내용도 달라질 수 있다.

### 6.3. 공통 실행 메서드

| 메서드 | 의미 | 결과를 받는 방식 |
| --- | --- | --- |
| `invoke()` | 단일 입력 실행 | 완성된 결과 하나 |
| `ainvoke()` | 비동기 단일 실행 | `await`로 결과 받기 |
| `batch()` | 여러 입력 실행 | 입력 순서의 결과 목록 |
| `abatch()` | 비동기 일괄 실행 | `await`로 결과 목록 받기 |
| `stream()` | 단일 입력의 결과 스트리밍 | 청크 순회 |
| `astream()` | 비동기 스트리밍 | `async for`로 청크 순회 |
| `batch_as_completed()` | 여러 입력의 완료 결과 순회 | 완료 순서로 `(입력 인덱스, 결과)` 받기 |

아래는 원본에서 표로 소개한 메서드를 직접 사용하는 **추가 예제**다.

```python
inputs = [
    {"topic": "RAG"},
    {"topic": "임베딩"},
    {"topic": "벡터 데이터베이스"},
]

answers = basic_chain.batch(inputs, config={"max_concurrency": 2})
for item, answer in zip(inputs, answers):
    print(item["topic"], ":", answer)
```

`batch()`는 여러 입력을 실행하고 입력 순서에 맞춰 결과를 돌려준다. `max_concurrency`는 동시 실행 수를 제한한다. 여러 입력이 하나의 프롬프트로 합쳐지는 것은 아니며, 제공자의 별도 Batch API와도 구별해야 한다.

```python
for index, answer in basic_chain.batch_as_completed(
    inputs,
    config={"max_concurrency": 2},
):
    print("완료된 입력:", inputs[index]["topic"])
    print(answer)
```

이 방식은 빠르게 끝난 결과부터 처리한다. 반환된 인덱스로 원래 입력을 찾을 수 있다.

다음 코드는 최상위 `await`를 지원하는 Jupyter Notebook에서 실행한다.

```python
answer = await basic_chain.ainvoke({"topic": "RAG"})
print(answer)

answers = await basic_chain.abatch(
    inputs,
    config={"max_concurrency": 2},
)
print(answers)

async for chunk in basic_chain.astream({"topic": "임베딩"}):
    print(chunk, end="", flush=True)
print()
```

일반 `.py` 파일에서는 해당 코드를 `async def main():` 안에 넣고 `asyncio.run(main())`으로 실행한다. 비동기는 기다리는 동안 다른 작업을 진행할 수 있게 하는 방식이며 모델 자체가 더 빨리 답하도록 하는 설정은 아니다.

공통 실행 규칙과 순차·병렬 조합은 [Runnable API 문서](https://reference.langchain.com/python/langchain-core/runnables/base)에 정의되어 있다.

### 6.4. `RunnablePassthrough`: 입력 그대로 전달하기

```python
from langchain_core.runnables import RunnablePassthrough

passthrough = RunnablePassthrough()
result = passthrough.invoke({"topic": "RAG"})

print(result)  # {'topic': 'RAG'}
```

입력을 그대로 다음 단계에 보낸다. 단독으로는 단순해 보이지만, 여러 갈래의 흐름에서 원본 입력을 유지하고 싶을 때 유용하다.

### 6.5. `.assign()`: 원본 딕셔너리에 값 추가하기

```python
upper = lambda x: x["word"].upper()

assign_chain = RunnablePassthrough.assign(upper=upper)
result = assign_chain.invoke({"word": "hello"})

print(result)  # {'word': 'hello', 'upper': 'HELLO'}
```

`upper` 함수는 입력 딕셔너리에서 `word`를 꺼내 대문자로 바꾼다. `.assign(upper=upper)`는 그 반환값을 결과의 `upper` 키에 넣는다. 기존의 `word`도 결과에 남는다.

`.assign()`의 입력은 딕셔너리여야 한다. 기존에 같은 이름의 키가 있으면 결과에서는 계산한 값으로 대체된다. 원본 변수에 직접 키를 추가하는 동작으로 이해하기보다는, 기존 필드와 새 필드를 합친 결과를 반환한다고 이해하면 좋다. [RunnablePassthrough API](https://reference.langchain.com/python/langchain-core/runnables/passthrough/RunnablePassthrough)

### 6.6. `RunnableLambda`: Python 함수 연결하기

```python
from langchain_core.runnables import RunnableLambda


def add_prefix(text: str) -> str:
    return f"[결과] {text}"


prompt = ChatPromptTemplate.from_messages([
    ("human", "{topic}이 뭔가요?"),
])

prefix_chain = (
    prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(add_prefix)
)

result = prefix_chain.invoke({"topic": "LangGraph"})
print(result)
```

`RunnableLambda`는 일반 함수를 체인에 연결할 수 있게 감싼다. `add_prefix()`는 문자열을 받으므로 문자열 파서 뒤에 둔다. 파서 없이 바로 모델 뒤에 연결하면 메시지 객체가 전달되어 의도와 다른 결과가 나올 수 있다.

입력 정리, 결과 형식 변경, 통계 계산처럼 Python으로 처리할 수 있는 로직을 이런 방식으로 추가할 수 있다. 다만 일반 `RunnableLambda`를 추가했다고 해서 함수가 토큰마다 실행되는 것은 아니다. 완성된 입력을 받아 처리하는 단계는 스트리밍 전달을 지연시킬 수 있다.

### 6.7. `RunnableParallel`: 여러 체인에 같은 입력 보내기

```python
from langchain_core.runnables import RunnableParallel

prompt_ko = ChatPromptTemplate.from_messages([
    ("human", "{topic}을 한국어로 설명해줘"),
])
prompt_en = ChatPromptTemplate.from_messages([
    ("human", "Explain {topic} in English"),
])

chain_ko = prompt_ko | llm | StrOutputParser()
chain_en = prompt_en | llm | StrOutputParser()

parallel_chain = RunnableParallel(
    korean=chain_ko,
    english=chain_en,
)

result = parallel_chain.invoke({"topic": "RAG"})

print("[한국어]", result["korean"])
print("[English]", result["english"])
```

두 체인 모두 `{"topic": "RAG"}`를 받는다. 실행 결과는 `korean`, `english` 키를 가진 딕셔너리로 모인다. 한국어 답변을 영어로 번역하는 순차 실행이 아니라, 같은 주제에 대해 각각 답변을 생성하는 병렬 실행이다.

```text
                       ┌─ 한국어 프롬프트 → 모델 → 문자열 ─ korean ┐
{"topic": "RAG"} ───────┤                                         ├─ 결과 딕셔너리
                       └─ 영어 프롬프트 → 모델 → 문자열 ─ english ┘
```

여기서는 한 번의 `parallel_chain.invoke()` 안에서 모델 호출이 두 번 일어난다. 지연 시간을 줄이는 데 도움이 될 수 있지만, 모델 사용량이 한 번으로 합쳐지는 것은 아니다. [RunnableParallel API](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableParallel)

## 7. Runnable 실습 문제: 응답의 단어 수 세기

05번 노트북의 마지막 문제는 LLM 응답을 받아 다음 형태로 반환하는 것이다.

```python
{"text": "모델이 생성한 설명", "word_count": 42}
```

원본 문제의 직접 모델 생성 코드를 공통 연결 함수로 통일하면 다음과 같다.

```python
from common_config import llm_connect
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

prompt = ChatPromptTemplate.from_template(
    "{topic}에 대해 초보자도 이해할 수 있게 설명해줘."
)
llm = llm_connect(model="gpt-5.4-mini")
parser = StrOutputParser()


def count_words(text: str) -> dict:
    return {
        "text": text,
        "word_count": len(text.split()),
    }


word_count_runnable = RunnableLambda(count_words)
word_count_chain = prompt | llm | parser | word_count_runnable

result = word_count_chain.invoke({"topic": "LangChain Runnable"})

print(result["word_count"])
print("-" * 20)
print(result["text"])
```

프롬프트가 질문을 만들고, 모델이 답변을 생성하고, 파서가 문자열을 만든다. 마지막으로 `count_words()`가 원문과 계산 결과를 함께 담은 딕셔너리를 반환한다.

`text.split()`은 공백을 기준으로 문자열을 나눈다. 여기서 계산하는 값은 공백으로 구분된 항목 수다. 한국어에서는 대체로 어절 수에 가까우며, 언어학적인 단어 수나 모델의 토큰 수와 일치하지 않는다.

원본 문제는 `ChatOpenAI(model="gpt-4o-mini", temperature=0)`를 직접 생성했다. 이 경우 앞에서 사용한 `LLM_API_KEY`와 `BASE_URL`이 그 생성자에 자동으로 전달되는 것은 아니다. 공통 함수를 사용하면 프로젝트의 연결 설정을 이어서 사용할 수 있다.

## 8. 실습하면서 구분해 두면 좋은 점

| 헷갈리기 쉬운 부분 | 이해해야 할 동작 |
| --- | --- |
| `max_tokens = 2000`을 선언하면 적용될까? | 연결 함수에 전달하지 않으면 함수 내부의 512 설정이 유지된다. |
| `invoke()`는 항상 모델을 호출할까? | 템플릿은 프롬프트를 만들고, 모델은 응답을 생성하고, 파서는 결과를 변환한다. |
| `MessagesPlaceholder`가 기억을 저장할까? | 전달받은 메시지를 삽입한다. 저장과 갱신은 별도로 처리한다. |
| `JsonOutputParser`면 필드도 검증할까? | JSON 변환과 필드 스키마 검증은 구분해야 한다. |
| `description="10점 만점"`이면 범위가 제한될까? | 범위 검증에는 `ge`, `le` 같은 제약이 필요하다. |
| `flush=True`이면 모델이 빨라질까? | Python의 출력 버퍼 처리만 바뀐다. |
| `RunnableParallel`이면 한 번만 과금될까? | 각 갈래의 모델 호출은 별도로 발생한다. |
| `len(text.split())`이 토큰 수일까? | 공백으로 나눈 항목 수다. 토큰 사용량과 다르다. |

오늘 실습에서 가장 많이 사용한 코드는 `prompt | llm | parser`였다. 각 단계가 받는 입력과 반환하는 출력을 이해하면 대화 기록, 구조화된 응답, Python 후처리도 같은 흐름으로 연결할 수 있다.

다음 글인 [프롬프트 엔지니어링 실습](./prompt_engineering_blog.md)에서는 같은 질문에 서로 다른 지시를 적용해 답변의 형식과 관점을 비교한다.

---

이 글은 `01_mono_test.ipynb`, `02_chat_model.ipynb`, `03_prompt_message.ipynb`, `04_output_parser.ipynb`, `05_runnable.ipynb`와 `common_config.py`를 바탕으로 재구성했다. 06번 Tool과 07번 Agent 실습은 포함하지 않았다.

태그: `Python`, `LangChain`, `ChatModel`, `LCEL`, `PromptTemplate`, `OutputParser`, `Runnable`
