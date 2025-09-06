# Durian AI
Durian AI is DataLake AI Agent

## 실행 방법

### 1. 의존성 설치
아래 명령어를 실행하여 프로젝트의 의존성을 설치합니다:
```bash
git clone https://github.com/yoongoing/durian-ai.git

uv sync
```

### 2. 환경 변수 설정
`durian-ai/.env` 파일을 생성하고, 발급받은 API 키로 아래와 같이 변경해줍니다.
```bash
DATABASE_URL=sqlite:///./datalake.db

OPENAI_API_KEY=본인키

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=본인키
LANGSMITH_PROJECT=DURIAN-AI

TAVILY_API_KEY=본인키

OPENROUTER_API_KEY=본인키
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### 3. 서버 실행
먼저 가상환경을 activate 해주고, 그 다음 실행해줍니다.
```bash
# MAC OS
source .venv/bin/activate

uvicorn app.main:app --reload
```

### 4. API 문서 확인
서버가 실행된 후, 브라우저에서 아래 URL로 이동하여 API 문서를 확인할 수 있습니다.

* Swagger UI: http://127.0.0.1:8000/docs
* ReDoc: http://127.0.0.1:8000/redoc


## 개발 스펙

이 프로젝트는 다음과 같은 주요 기술 스택과 버전을 기반으로 개발되었습니다:

- **Python**: 3.11
- **FastAPI**: 최신 버전 (권장: 0.100.0 이상)
- **LangGraph**: 0.2.5
- **LangChain**: 0.0.304
- **Pydantic**: 2.0 이상
- **Uvicorn**: 0.22.0

추가적으로, 환경 변수 관리를 위해 `.env` 파일을 사용하며, 기본 데이터베이스로 SQLite를 사용합니다.