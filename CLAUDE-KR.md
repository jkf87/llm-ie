# CLAUDE.md

이 파일은 이 리포지토리의 코드로 작업할 때 Claude Code(claude.ai/code)에 지침을 제공합니다.

## 프로젝트 개요

LLM-IE는 LLM 기반 정보 추출을 위한 포괄적인 툴킷으로, **개체명 인식**, **속성 추출**, **관계 추출** 파이프라인을 위한 구성 요소를 제공합니다. 이 프로젝트는 세 가지 주요 구성 요소로 이루어져 있습니다.

1.  **핵심 파이썬 패키지** (`package/llm-ie/`): 추출기, 청커, 엔진, 데이터 타입을 포함하는 메인 라이브러리
2.  **웹 애플리케이션** (`web_app/`): LLM-IE 기능을 드래그 앤 드롭으로 사용할 수 있는 Flask 기반 UI
3.  **문서** (`docs/`): MkDocs 기반의 문서 사이트

-----

## 아키텍처 개요

### 핵심 구성 요소

  - **Engines** (`engines.py`): 여러 LLM 제공업체(OpenAI, Ollama, vLLM, HuggingFace 등)를 지원하는 추상 추론 엔진 인터페이스
  - **Extractors** (`extractors.py`): 다양한 프롬프팅 전략을 사용하는 프레임 및 관계 추출 클래스:
      - `DirectFrameExtractor`: 단일 패스 추출
      - `ReviewFrameExtractor`: 검토 단계가 포함된 다중 패스
      - `AttributeExtractor`: 엔티티 속성 추출
      - `BinaryRelationExtractor`/`MultiClassRelationExtractor`: 관계 추출
  - **Chunkers** (`chunkers.py`): 텍스트 분할 전략:
      - `UnitChunker`: 처리 단위(문장, 문서, 줄) 정의
      - `ContextChunker`: 단위 주변의 컨텍스트 제공(슬라이딩 윈도우, 전체 문서)
  - **Data Types** (`data_types.py`): 프레임 및 문서를 위한 핵심 데이터 구조
  - **Prompt Editor** (`prompt_editor.py`): 프롬프트 엔지니어링을 위한 대화형 LLM 에이전트

### 처리 파이프라인

1.  **텍스트 청킹**: 문서는 선택적 컨텍스트와 함께 단위(문장/단락)로 분할됩니다.
2.  **LLM 추론**: 각 단위는 구성된 추출 파이프라인을 통해 처리됩니다.
3.  **프레임 수집**: 추출된 엔티티/관계는 `LLMInformationExtractionFrame` 객체로 수집됩니다.
4.  **문서 조립**: 프레임은 관리 및 시각화를 위해 `LLMInformationExtractionDocument`로 집계됩니다.

-----

## 개발 명령어

### 파이썬 패키지 개발

```bash
# 패키지 디렉토리로 이동
cd package/llm-ie/

# 의존성 설치 (Poetry 사용)
poetry install

# 패키지 빌드
poetry build

# 개발 모드로 설치
pip install -e .
```

### 웹 애플리케이션 개발

```bash
# 웹 앱 디렉토리로 이동
cd web_app/

# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
python run.py

# Gunicorn으로 실행 (프로덕션)
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

### 문서

```bash
# 로컬에서 문서 제공
mkdocs serve

# 문서 빌드
mkdocs build

# GitHub Pages에 배포
mkdocs gh-deploy
```

### 도커 (웹 앱)

```bash
# 이미지 빌드
docker build -t llm-ie-web-app web_app/

# 컨테이너 실행
docker run -p 5000:5000 llm-ie-web-app
```

-----

## 주요 설정 파일

  - `package/llm-ie/pyproject.toml`: Poetry를 사용한 파이썬 패키지 설정
  - `web_app/requirements.txt`: 웹 애플리케이션 의존성
  - `mkdocs.yml`: 문서 사이트 설정
  - `package/llm-ie/pipelines/sample_config.yaml`: 샘플 파이프라인 설정

-----

## LLM 엔진 설정

이 프로젝트는 통합된 인터페이스를 통해 여러 LLM 제공업체를 지원합니다. 각 엔진은 특정 설정이 필요합니다.

  - **OpenAI**: `OPENAI_API_KEY` 환경 변수 필요
  - **Azure OpenAI**: 엔드포인트 및 API 버전 설정 필요
  - **Ollama**: 로컬 Ollama 인스턴스에 연결
  - **vLLM**: OpenAI 호환 서버 인터페이스 사용
  - **HuggingFace**: 허브를 통한 직접 모델 로딩

모든 엔진은 추론 모델(o3, Qwen3)을 위해 `BasicLLMConfig`와 `ReasoningLLMConfig`를 모두 지원합니다.

-----

## 파이프라인 처리

배치 처리를 위해서는 파이프라인 스크립트를 사용하세요:

  - `package/llm-ie/pipelines/sequential_frame_extraction.py`: 단일 스레드 처리
  - `package/llm-ie/pipelines/multithread_frame_extraction.py`: 다중 스레드 처리

샘플 형식을 따라 YAML 파일에서 파이프라인 매개변수를 설정하세요.

-----

## 에셋 관리

  - 기본 프롬프트: `package/llm-ie/src/llm_ie/asset/default_prompts/`
  - 프롬프트 가이드: `package/llm-ie/src/llm_ie/asset/prompt_guide/`
  - PromptEditor 프롬프트: `package/llm-ie/src/llm_ie/asset/PromptEditor_prompts/`

-----

## 웹 애플리케이션 구조

  - `app/routes.py`: Flask 라우트 정의
  - `app/extractors.py`: 웹 전용 추출 로직
  - `app/templates/`: UI 컴포넌트를 위한 Jinja2 템플릿
  - `app/static/`: CSS, JavaScript, 이미지 에셋