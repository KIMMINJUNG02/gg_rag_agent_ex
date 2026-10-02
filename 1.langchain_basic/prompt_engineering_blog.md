# LangChain 프롬프트 엔지니어링: 여섯 가지 기법과 비교 실험

> 학습일: 2026년 10월 1일  
> 실습 자료: `08_prompt_engineering.ipynb`

같은 모델에 같은 질문을 하더라도 지시와 예시, 역할, 출력 형식을 어떻게 정하느냐에 따라 답변이 달라진다. 이 글에서는 Zero-shot, Few-shot, 단계별 분석, 역할 프롬프트, 구조화된 출력, 프롬프트 팩토리를 코드로 살펴본다.

모델 호출과 LCEL의 기초는 [모델 호출과 체인 구성](./model_call_blog.md)에 정리했다. 이 글은 별도 노트북에서도 따라 할 수 있도록 필요한 연결 설정부터 포함한다. 예제는 위에서 아래로 같은 커널에서 실행하며, 원본에서 보완한 부분은 해당 위치에 표시했다.

## 1. 실습 준비

이 프로젝트에서 확인한 환경은 Python 3.12 이상, `langchain 1.4.3`, `langchain-core 1.6.6`, `langchain-openai 1.6.7`이다. 프로젝트의 설치 버전이며 최신 버전을 뜻하지 않는다.

저장소를 사용한다면 프로젝트 루트에서 다음 명령으로 환경을 준비하고 해당 환경을 노트북 커널로 선택한다.

```bash
uv sync --locked
```

새로운 환경에서는 다음 패키지를 설치할 수 있다. 이 명령은 버전을 고정하지 않는다.

```bash
python -m pip install langchain langchain-openai python-dotenv ipykernel
```

프로젝트 루트의 `.env`에는 발급받은 연결 정보를 입력한다. 아래는 자리표시자이며 실제 키는 공개하지 않는다.

```dotenv
LLM_API_KEY=여기에_발급받은_API_키
BASE_URL=여기에_제공받은_API_기본_URL
```

다음 코드는 현재 작업 디렉터리가 `1.langchain_basic`인 노트북을 기준으로 한다. 프로젝트 루트에서 실행한다면 `dotenv_path=".env"`로 바꾼다.

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True, dotenv_path="../.env")


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

`load_dotenv()`는 연결 정보를 환경 변수로 읽고, `override=True`는 같은 이름의 기존 값을 덮어쓴다. `os.environ[...]`은 필수 환경 변수가 빠져 있으면 오류를 낸다. 함수는 연결 객체를 만들며 실제 요청은 이후 `invoke()`에서 발생한다.

이 함수는 프로젝트의 공통 연결 설정을 바탕으로 토큰 예산을 인자로 받도록 만든 보완 예제다. `use_responses_api=False`는 수업의 MonoRouter 설정을 따른 것이며, 모든 API 서비스에 필수인 설정은 아니다. 모델 식별자와 생성 매개변수 지원 여부는 사용하는 라우터에서 확인해야 한다. `temperature=0`이어도 완전히 동일한 답변을 보장하지 않는다.

## 2. 프롬프트 엔지니어링 — 08번 노트북

LangChain이 처리 흐름을 구성한다면, 프롬프트 엔지니어링은 그 흐름 안에서 모델에 어떤 지시를 줄지 다룬다. 이번에는 같은 질문을 여러 방식으로 전달해 응답의 구조와 관점이 어떻게 달라지는지 살펴본다.

공통 질문은 “재택근무 전면 도입, 찬성해야 할까요?”다. 실제 회사의 데이터를 넣지 않았으므로 아래 답변은 프롬프트 실습 결과로 해석한다.

08번 원본도 `ChatOpenAI(model="gpt-4o-mini", temperature=0)`를 직접 사용한다. 이 글에서는 실습 준비에서 정의한 `llm_connect_custom()`으로 연결을 통일하고, 긴 응답을 비교할 수 있도록 토큰 예산을 늘린다. 모델과 생성 설정을 바꿨으므로 원본 출력과 동일한 결과를 기대하는 실험은 아니다.

```python
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = llm_connect_custom(
    model="gpt-5.4-mini",
    temperature=0,
    max_tokens=2000,
)

QUESTION = "재택근무 전면 도입, 찬성해야 할까요?"
```

### 2.1. Zero-shot: 예시 없이 질문하기

```python
zero_shot_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
])

zero_shot_chain = zero_shot_prompt | llm | StrOutputParser()
result = zero_shot_chain.invoke({"question": QUESTION})

print("=== Zero-shot 결과 ===")
print(result)
```

Zero-shot은 원하는 답변의 예시를 주지 않고 작업을 요청하는 방식이다. 코드가 간단해서 첫 결과를 확인하기 좋다. 다만 출력 형식을 별도로 정하지 않았으므로 목록, 문단, 결론의 구성은 달라질 수 있다.

Zero-shot이 지시나 문맥이 전혀 없다는 뜻은 아니다. 핵심은 작업의 입출력 예시를 제공하지 않았다는 점이다.

### 2.2. Few-shot: 예시로 형식 보여 주기

```python
few_shot_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """아래 예시와 똑같은 형식으로 답변하세요.

예시 1)
질문: 주 4일 근무제를 도입해야 할까요?
답변:
✅ 찬성 근거
- 직원 삶의 질 향상으로 생산성 증가
- 채용 경쟁력 강화

❌ 반대 근거
- 업무량 미감소 시 오히려 번아웃 위험
- 서비스 연속성 유지 어려움

💡 결론: 업종·직무별로 차등 적용하는 파일럿 테스트를 권장합니다.

예시 2)
질문: 사내 카페테리아를 없애야 할까요?
답변:
✅ 찬성 근거
- 운영 비용 절감
- 외부 다양한 식사 옵션 활용 가능

❌ 반대 근거
- 직원 식사 복지 감소
- 점심 시간 이탈로 협업 기회 감소

💡 결론: 직원 수요 조사 후 결정하되, 식대 지원으로 보완을 고려하세요.
""",
    ),
    ("human", "질문: {question}"),
])

few_shot_chain = few_shot_prompt | llm | StrOutputParser()
result = few_shot_chain.invoke({"question": QUESTION})

print("=== Few-shot 결과 ===")
print(result)
```

두 예시는 `찬성 근거 → 반대 근거 → 결론`이라는 형식을 보여 준다. 모델은 현재 프롬프트에 들어 있는 예시를 참고해 새 질문에도 유사한 형식으로 답하도록 유도된다. 모델의 가중치를 새로 학습시키는 파인튜닝은 아니다.

예시의 내용도 답변에 영향을 줄 수 있다. 일관된 형식을 원한다면 서로 모순되지 않는 예시를 선택하고, 필요한 수만 넣는 것이 좋다. 예시가 늘어나면 입력 토큰도 늘어난다.

### 2.3. Chain-of-Thought 실습: 분석 항목을 나누어 요청하기

원본은 문제 파악, 이해관계자 분석, 장단점 검토, 최종 판단의 네 단계로 설명하도록 요청한다. 아래는 그 지시를 유지한 코드다.

```python
cot_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """질문에 답하기 전에 반드시 다음 4단계로 생각하고, 각 단계를 명시하세요.

[1단계: 문제 파악] 무엇을 묻고 있는가?
[2단계: 이해관계자 분석] 누가 영향을 받는가?
[3단계: 장단점 검토] 도입/미도입 시 각각 어떤 결과가?
[4단계: 최종 판단] 근거를 바탕으로 한 결론은?""",
    ),
    ("human", "{question}"),
])

cot_chain = cot_prompt | llm | StrOutputParser()
result = cot_chain.invoke({"question": QUESTION})

print("=== 단계별 분석 결과 ===")
print(result)
```

이 실습에서는 답변에 포함할 분석 항목을 명시해 빠뜨리는 부분을 줄이고자 한다. 다만 생성된 단계별 설명은 사용자에게 보여 주는 설명문이며, 모델의 내부 추론 전체를 그대로 확인한 것으로 볼 수 없다. 응답이 길거나 단계가 많다고 정확성이 보장되는 것도 아니다.

실제 활용에서는 “문제 정의, 주요 장단점, 결론과 핵심 근거를 간결하게 제시해 주세요”처럼 확인 가능한 설명을 요청하는 방법도 있다. 기법의 효과는 질문과 모델에 따라 달라지므로 결과를 비교해야 한다.

### 2.4. 역할 프롬프트: 같은 질문에 다른 관점 부여하기

```python
def make_role_chain(role_description: str):
    """역할 설명을 받아 해당 관점의 응답 체인을 만든다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", role_description),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


hr_chain = make_role_chain("""
당신은 15년 경력의 HR 디렉터입니다.
직원 복지, 조직 문화, 인재 유지에 최우선 가치를 둡니다.
답변은 구성원 관점에서 3~4문장으로 핵심만 짚어주세요.
""")

cfo_chain = make_role_chain("""
당신은 10년 경력의 재무이사(CFO)입니다.
모든 결정을 비용-효익 분석으로 접근합니다.
답변은 비용과 효과를 중심으로 3~4문장으로 핵심만 짚어주세요.
제공된 수치가 없으면 임의로 만들지 말고 필요한 지표를 제시하세요.
""")

print("=== HR 디렉터 관점 ===")
print(hr_chain.invoke({"question": QUESTION}))

print("\n=== CFO 관점 ===")
print(cfo_chain.invoke({"question": QUESTION}))
```

역할 설명을 함수 인자로 받으므로 질문 형식과 연결 코드는 재사용할 수 있다. HR은 구성원 경험을, CFO는 비용과 효과를 중심으로 답하도록 지시했다. 원본 CFO 예제의 “수치 근거” 요청은 근거 없는 숫자를 만들지 않도록 보완했다.

“15년 경력”이라는 역할을 부여한다고 실제 전문 자격이나 사실성이 생기는 것은 아니다. 역할 프롬프트는 관점과 표현 방식을 조절하는 수단이다.

원본에는 docstring과 데코레이터를 연결한 주석이 있지만, 여기서 docstring은 Python 함수 설명일 뿐 자동으로 모델에 전달되지 않는다. 모델이 받는 역할 지시는 `("system", role_description)`에 들어간 문자열이다.

### 2.5. 구조화된 출력: JSON으로 받아 활용하기

08번 노트북은 문자열 파서로 응답을 받고 `json.loads()`로 직접 변환한다. [모델 호출과 체인 구성](./model_call_blog.md)에서 다룬 `JsonOutputParser`와 비교하면 파싱 단계를 직접 작성한다는 차이가 있다.

아래 코드는 원본의 출력 필드 안내를 완전한 JSON 예시로 정리한 버전이다.

```python
structured_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """아래 구조의 JSON 객체 하나만 반환하세요.
Markdown 코드 블록이나 다른 설명을 포함하지 마세요.
confidence에는 1~10 사이의 정수를 넣으세요.

{{
  "topic": "주제",
  "pros": ["찬성 근거1", "찬성 근거2", "찬성 근거3"],
  "cons": ["반대 근거1", "반대 근거2"],
  "recommendation": "최종 권고사항",
  "confidence": 7
}}""",
    ),
    ("human", "다음 주제를 분석해줘: {question}"),
])

structured_chain = structured_prompt | llm | StrOutputParser()
raw_output = structured_chain.invoke({"question": "재택근무 전면 도입"})

print("=== LLM 원본 출력 ===")
print(raw_output)

try:
    data = json.loads(raw_output)
    print("주제:", data["topic"])
    print("확신도:", data["confidence"])
    print("권고사항:", data["recommendation"])
    print("찬성 근거:", ", ".join(data["pros"]))
except json.JSONDecodeError as error:
    print("JSON 파싱 실패:", error)
except (KeyError, TypeError) as error:
    print("필드 또는 데이터 타입 확인 필요:", error)
```

템플릿 안에서 실제 중괄호를 표현할 때는 `{{`, `}}`로 이스케이프한다. `{question}`처럼 한 쌍으로 쓰면 템플릿 변수로 해석되기 때문이다. 위 코드가 모델에 전달될 때 JSON 예시에는 일반 중괄호 한 쌍이 들어간다.

`json.loads()`는 JSON 문법을 읽는다. 유효한 JSON이어도 필요한 키가 없거나 값의 타입이 틀릴 수 있어 필드 사용 중의 오류도 처리했다. 다만 이 예제는 모든 필드와 범위를 검증하는 코드는 아니다. 전체 스키마 검증이 필요하면 [모델 호출과 체인 구성](./model_call_blog.md)의 Pydantic 모델을 적용할 수 있다.

프롬프트에 “반드시”라고 적는 것은 형식을 유도하는 지시다. 제공자 차원의 구조화 출력 기능과 동일한 보장은 아니다. 또한 `confidence`는 모델이 생성한 자기평가 수치이며, 실제 정확도나 보정된 확률로 해석하면 안 된다.

### 2.6. 프롬프트 팩토리: 같은 구조의 체인 재사용하기

```python
def make_analyst_chain(
    domain: str,
    perspective: str,
    output_format: str,
):
    """분야, 관점, 출력 형식을 받아 분석 체인을 만든다."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "당신은 {domain} 분야 전문 분석가입니다.\n"
            "{perspective} 관점에서 분석하세요.\n\n"
            "출력 형식:\n{output_format}",
        ),
        ("human", "{question}"),
    ]).partial(
        domain=domain,
        perspective=perspective,
        output_format=output_format,
    )
    return prompt | llm | StrOutputParser()


biz_chain = make_analyst_chain(
    domain="경영",
    perspective="ROI와 생산성 중심",
    output_format="[핵심 지표] → [예상 효과] → [실행 권고] (각 2~3줄)",
)

psych_chain = make_analyst_chain(
    domain="조직심리",
    perspective="구성원 동기·몰입 중심",
    output_format="[심리적 영향] → [리스크] → [권고사항] (각 2~3줄)",
)

print("=== 경영 분석 관점 ===")
print(biz_chain.invoke({"question": QUESTION}))

print("\n=== 조직심리 관점 ===")
print(psych_chain.invoke({"question": QUESTION}))
```

팩토리 함수는 설정을 받아 사용할 객체를 만들어 주는 함수다. 여기서는 세 가지 설정을 받아 분석 체인을 만든다.

원본은 Python f-string으로 시스템 메시지를 먼저 완성했다. 위 코드는 이를 `.partial()`로 바꾼 **보완 예제**다. `domain`, `perspective`, `output_format`을 미리 채워 두고, 실행 시에는 `question`만 받는다. 출력 형식 문자열에 JSON 중괄호가 포함되더라도 그 값을 다시 템플릿 문법으로 해석하는 문제를 피할 수 있다.

역할 프롬프트, Few-shot, JSON 지시는 함께 사용할 수도 있다. 팩토리는 그런 지시를 반복해서 구성하기 위한 코드 재사용 방법이므로 다른 기법과 배타적인 선택지가 아니다.

## 3. 같은 질문으로 기법 비교하기

08번 노트북은 여섯 가지 기법을 소개하지만, 마지막 비교 반복문에서는 Zero-shot, Few-shot, CoT 세 가지를 비교한다. 아래도 같은 범위를 따른다.

```python
techniques = [
    ("Zero-shot", zero_shot_prompt),
    ("Few-shot", few_shot_prompt),
    ("Chain-of-Thought", cot_prompt),
]

print("질문:", QUESTION)
print("=" * 60)

results = {}
for name, prompt in techniques:
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": QUESTION})
    results[name] = answer

print(f"\n{'기법':<20} {'글자수':>8} {'줄 수':>6}")
print("-" * 38)

for name, answer in results.items():
    print(f"{name:<20} {len(answer):>8,} {len(answer.splitlines()):>6}")
```

같은 모델과 질문을 사용하고 프롬프트만 바꾸어 결과를 모은다. `len(answer)`는 Python 문자열 길이를, `splitlines()`는 줄 단위로 나눈 목록을 반환한다. 따라서 출력 표는 응답의 길이와 줄 수를 비교하는 자료다.

```python
for name, answer in results.items():
    print(f"\n{'=' * 60}")
    print(f"[{name}]")
    print("=" * 60)

    preview = answer[:400]
    print(preview)

    if len(answer) > 400:
        print(f"\n... (총 {len(answer):,}자 중 400자 미리보기)")
```

원본 주석에는 “답변 전체 출력”이라고 되어 있지만 실제 코드는 처음 400자만 보여 준다. 전체를 비교하려면 `print(answer)`를 사용한다.

비교할 때는 길이 외에도 질문에 답했는지, 지정한 형식을 지켰는지, 근거 없는 수치를 만들지 않았는지, 결론과 근거가 연결되는지를 살펴본다. 응답이 길다는 사실만으로 더 좋은 답변이라고 판단할 수 없다. 이 글에는 실행하지 않은 모델 응답이나 실측 비교 수치를 넣지 않았다.

## 4. 기법을 적용할 때 확인할 점

| 헷갈리기 쉬운 부분 | 이해해야 할 동작 |
| --- | --- |
| Few-shot이 모델 재학습일까? | 현재 요청의 예시를 참고하는 방식이다. |
| 역할을 부여하면 전문가가 될까? | 답변 관점을 유도하지만 전문성과 사실성을 보장하지 않는다. |
| JSON만 요청하면 항상 파싱될까? | 형식 오류나 잘림이 가능하므로 파싱·검증 처리가 필요하다. |
| 단계별 설명이 길면 정확할까? | 설명의 길이와 정확성은 별개이며, 생성된 설명이 내부 추론 전체를 뜻하지 않는다. |
| 팩토리는 다른 기법과 별개로만 쓸까? | 체인을 재사용하는 코드 구성 방식이므로 역할·예시·출력 형식 지시를 함께 담을 수 있다. |
| 글자 수와 줄 수로 품질을 판단할까? | 형식 준수, 사실성, 근거와 결론의 연결까지 함께 살펴본다. |

이 실습의 목적은 지시가 결과에 미치는 영향을 비교하는 것이다. 질문과 모델을 고정하고 출력 형식이나 관점을 바꾸면서, 원하는 작업에 맞는 프롬프트를 찾아갈 수 있다.

---

이 글은 `08_prompt_engineering.ipynb`를 바탕으로 재구성했다. 모델 연결과 토큰 설정을 프로젝트 환경에 맞췄고, JSON 예시와 프롬프트 팩토리 등 보완한 부분은 본문에 표시했다. 실제 API 응답이나 실측 비교 수치를 생성해 넣지는 않았다.

태그: `Python`, `LangChain`, `프롬프트엔지니어링`, `ZeroShot`, `FewShot`, `ChatPromptTemplate`, `구조화된출력`
