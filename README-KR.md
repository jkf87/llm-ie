<div align="center"><img src=docs/readme_img/LLM-IE.png width=500 ></div>

![Python Version](https://img.shields.io/pypi/pyversions/llm-ie)
![PyPI](https://img.shields.io/pypi/v/llm-ie)
[![Paper](https://img.shields.io/badge/DOI-10.1093/jamiaopen/ooaf012-red)](https://doi.org/10.1093/jamiaopen/ooaf012)
[![Website](https://img.shields.io/badge/website-GitHub.io-purple)](https://daviden1013.github.io/llm-ie/)

**[English](README.md) | 한국어**

LLM 기반 개체명 인식, 속성 추출 및 관계 추출 파이프라인을 위한 구성 요소를 제공하는 포괄적인 툴킷입니다.

| 기능 | 지원 |
|------|------|
| **프롬프트 작성을 위한 LLM 에이전트** | :white_check_mark: 웹 앱, 대화형 채팅 |
| **개체명 인식 (NER)** | :white_check_mark: 사용자 정의 가능한 세분화 (예: 문장 수준, 문서 수준) |
| **개체 속성 추출** | :white_check_mark: 유연한 형식 |
| **관계 추출 (RE)** | :white_check_mark: 이진 및 다중 클래스 관계 |
| **시각화** | :white_check_mark: 웹 앱, 내장 개체 및 관계 시각화 |
| **배치 처리** | :white_check_mark: Gemini Direct API, 다중 API 키 지원 |

## 🆕최근 업데이트
- [v1.0.0](https://github.com/daviden1013/llm-ie/releases/tag/v1.0.0) (2025년 5월 15일): 
  - 📐**사용자 가이드**가 [문서 페이지](https://daviden1013.github.io/llm-ie/)로 이동되었습니다.
  - **웹 애플리케이션**이 *LLM-IE*에 드래그 앤 드롭 액세스를 제공합니다.
  - 청킹 방법(예: 문장)과 프롬프팅 방법(예: 리뷰)을 분리하여 **`FrameExtractor` 리팩토링**. 청킹은 이제 `UnitChunker`와 `ContextChunker`에서 정의되고, `FrameExtractor`는 프롬프팅 방법을 정의합니다.
  - **문서 웹사이트**. 사용자 가이드와 API 참조가 이제 제공됩니다.
  - **최적화된 동시/배치 처리**. 계산 리소스를 더 잘 활용하기 위해 세마포어를 채택했습니다.
- [v1.1.0](https://github.com/daviden1013/llm-ie/releases/tag/v1.1.0) (2025년 5월 19일): 추론 모델(o3, Qwen3)을 지원하는 LLM별 구성.
- [v1.2.0](https://github.com/daviden1013/llm-ie/releases/tag/v1.2.0) (2025년 6월 15일): 복잡한 속성 스키마를 위한 속성 추출기.
- [v1.2.1](https://github.com/daviden1013/llm-ie/releases/tag/v1.2.1) (2025년 7월 12일): 프롬프트 에디터에 채팅 기록 내보내기/가져오기 기능 추가.
- [v1.2.2](https://github.com/daviden1013/llm-ie/releases/tag/v1.2.2) (2025년 8월 25일): 추론 LLM(GPT-OSS, Qwen3)용 구성 추가.
- **배치 처리 기능 추가** (2025년 9월): Gemini Direct API 통합 및 고급 배치 처리 기능 추가.

## 📑목차
- [개요](#개요)
- [전제 조건](#전제-조건)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [웹 애플리케이션](#웹-애플리케이션)
- [배치 처리](#배치-처리)
- [예제](#예제)
- [유용한 스크립트](#유용한-스크립트)
- [사용자 가이드](#사용자-가이드)
- [벤치마크](#벤치마크)
- [인용](#인용)

## ✨개요
LLM-IE는 개체명, 개체 속성 및 개체 관계 추출을 위한 강력한 정보 추출 유틸리티를 제공하는 툴킷입니다. 아래 플로우차트는 일반적인 언어 요청부터 출력 시각화까지의 워크플로우를 보여줍니다.

<div align="center"><img src="docs/readme_img/LLM-IE flowchart.png" width=800 ></div>

## 🚦전제 조건
최소 하나의 LLM 추론 엔진이 필요합니다. 🚅 [LiteLLM](https://github.com/BerriAI/litellm), 🦙 [Llama-cpp-python](https://github.com/abetlen/llama-cpp-python), <img src="docs/readme_img/ollama_icon.png" alt="Icon" width="22"/> [Ollama](https://github.com/ollama/ollama), 🤗 [Huggingface_hub](https://github.com/huggingface/huggingface_hub), <img src=docs/readme_img/openai-logomark_white.png width=16 /> [OpenAI API](https://platform.openai.com/docs/api-reference/introduction), <img src=docs/readme_img/vllm-logo_small.png width=20 /> [vLLM](https://github.com/vllm-project/vllm), 그리고 **Gemini Direct API**에 대한 내장 지원이 있습니다. 설치 가이드는 해당 프로젝트를 참조하세요. 다른 추론 엔진은 [InferenceEngine](src/llm_ie/engines.py) 추상 클래스를 통해 구성할 수 있습니다.

## 💿설치
Python 패키지는 PyPI에서 사용할 수 있습니다.
```bash
pip install llm-ie 
```
이 패키지는 LLM 추론 엔진 설치를 확인하거나 설치하지 않습니다. 자세한 내용은 [전제 조건](#전제-조건) 섹션을 참조하세요.

## 🚀빠른 시작
ChatGPT로 합성된 [의료 노트](demo/document/synthesized_note.txt)를 사용하여 정보 추출 과정을 데모합니다. 우리의 작업은 진단명, 범위 및 해당 속성(즉, 진단 날짜, 상태)을 추출하는 것입니다.

### LLM 추론 엔진 선택
아래의 내장 엔진 중 하나를 선택하세요.

<details>
<summary>🚅 LiteLLM</summary>

```python
from llm_ie.engines import LiteLLMInferenceEngine

inference_engine = LiteLLMInferenceEngine(model="openai/Llama-3.3-70B-Instruct", base_url="http://localhost:8000/v1", api_key="EMPTY")
```
</details>

<details>
<summary><img src=docs/readme_img/openai-logomark_white.png width=16 /> OpenAI API 및 호환 서비스</summary>

[API 키 안전을 위한 모범 사례](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)를 따라 API 키를 설정하세요.
```python
from llm_ie.engines import OpenAIInferenceEngine

inference_engine = OpenAIInferenceEngine(model="gpt-4o-mini")
```

OpenAI 호환 서비스(예: OpenRouter)의 경우:
```python
from llm_ie.engines import OpenAIInferenceEngine

inference_engine = OpenAIInferenceEngine(base_url="https://openrouter.ai/api/v1", model="meta-llama/llama-4-scout")
```

</details>

<details>
<summary><img src=docs/readme_img/Azure_icon.png width=32 /> Azure OpenAI API</summary>

[Azure AI Services 빠른 시작](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Ckeyless%2Ctypescript-keyless%2Cpython-new&pivots=programming-language-python)을 따라 엔드포인트와 API 키를 설정하세요.

```python
from llm_ie.engines import AzureOpenAIInferenceEngine

inference_engine = AzureOpenAIInferenceEngine(model="gpt-4o-mini", 
                                              api_version="<your api version>")
```

</details>

<details>
<summary>🤗 Huggingface_hub</summary>

```python
from llm_ie.engines import HuggingFaceHubInferenceEngine

inference_engine = HuggingFaceHubInferenceEngine(model="meta-llama/Meta-Llama-3-8B-Instruct")
```
</details>

<details>
<summary><img src="docs/readme_img/ollama_icon.png" alt="Icon" width="22"/> Ollama</summary>

```python 
from llm_ie.engines import OllamaInferenceEngine

inference_engine = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")
```
</details>

<details>
<summary><img src=docs/readme_img/vllm-logo_small.png width=20 /> vLLM</summary>

vLLM 지원은 [OpenAI 호환 서버](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)를 따릅니다. 더 많은 매개변수에 대해서는 문서를 참조하세요.

서버 시작
```cmd
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct
```
추론 엔진 정의
```python
from llm_ie.engines import OpenAIInferenceEngine
inference_engine = OpenAIInferenceEngine(base_url="http://localhost:8000/v1",
                                         api_key="EMPTY",
                                         model="meta-llama/Meta-Llama-3.1-8B-Instruct")
```
</details>

<details>
<summary>🦙 Llama-cpp-python</summary>

```python
from llm_ie.engines import LlamaCppInferenceEngine

inference_engine = LlamaCppInferenceEngine(repo_id="bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF",
                                           gguf_filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
```
</details>

이 빠른 시작 데모에서는 OpenRouter를 사용하여 프롬프트 엔지니어링에는 Llama-4-Scout를, 개체 및 속성 추출에는 Llama-3.1-70B-Instruct를 실행합니다.
다른 추론 엔진, LLM 또는 양자화로 인해 출력이 약간 다를 수 있습니다.

### LLM 에이전트와 채팅하여 프롬프트 엔지니어링
프롬프트 에디터 LLM 에이전트를 정의하는 것부터 시작합니다. OpenRouter API 키를 환경 변수 `OPENROUTER_API_KEY`에 저장합니다.
```bash
export OPENROUTER_API_KEY=<OpenRouter API 키>
```

```python
from llm_ie import OpenAIInferenceEngine, BasicLLMConfig, DirectFrameExtractor, PromptEditor, SentenceUnitChunker, SlideWindowContextChunker

# 프롬프트 에디터를 위한 LLM 추론 엔진 정의
prompt_editor_llm = OpenAIInferenceEngine(base_url="https://openrouter.ai/api/v1", 
                                          model="meta-llama/llama-4-scout", 
                                          api_key=os.getenv("OPENROUTER_API_KEY"),
                                          config=BasicLLMConfig(temperature=0.4, 
                                                                max_new_tokens=4096))
# LLM 프롬프트 에디터 정의
editor = PromptEditor(prompt_editor_llm, DirectFrameExtractor)
# 채팅 시작
editor.chat()
```

이것은 대화형 세션을 엽니다:
<div align="left"><img src=docs/readme_img/terminal_chat.PNG width=1000 ></div>

에이전트는 ```DirectFrameExtractor```에서 요구하는 스키마를 따라 프롬프트 템플릿을 작성합니다.
몇 번의 채팅 후, 시작할 프롬프트 템플릿을 얻습니다:

```
### 작업 설명
아래 단락에는 진단 목록이 포함된 임상 노트가 있습니다. 이를 주의 깊게 검토하고 진단 날짜와 상태를 포함한 진단을 추출하세요.

### 스키마 정의
출력에는 다음이 포함되어야 합니다:
    "entity_text"는 텍스트에 나타나는 진단명,
    "Date"는 진단이 내려진 날짜,
    "Status"는 진단의 현재 상태(예: 활성, 해결됨 등)

### 출력 형식 정의
출력은 JSON 형식을 따라야 합니다. 예를 들어:
[
    {"entity_text": "<진단명>", "attr": {"Date": "<YYYY-MM-DD 형식의 날짜>", "Status": "<상태>"}},
    {"entity_text": "<진단명>", "attr": {"Date": "<YYYY-MM-DD 형식의 날짜>", "Status": "<상태>"}}
]

### 추가 힌트
- 출력은 제공된 내용을 100% 기반으로 해야 합니다. 가짜 정보를 출력하지 마세요.
- 특정 날짜나 상태가 없는 경우 해당 키를 생략하세요.

### 컨텍스트
아래 텍스트는 임상 노트에서 가져온 것입니다:
"{{input}}"
```

### 정보 추출을 위한 프롬프팅 알고리즘 설계
LLM에 전체 문서를 프롬프트하는 대신(우리 실험에 따르면 성능이 더 나쁩니다), 입력 문서를 단위(예: 문장, 텍스트 줄, 단락)로 나눕니다. LLM은 다음 단위로 이동하기 전에 한 번에 하나의 단위에만 집중합니다. 이는 `UnitChunker` 클래스로 달성됩니다. 이 데모에서는 문장별 프롬프팅을 위해 `SentenceUnitChunker`를 사용합니다. LLM이 한 번에 하나의 문장에만 집중하지만, 이 경우 2개 문장의 슬라이드 윈도우를 컨텍스트로 제공합니다. 이는 LLM에 추가 정보를 제공합니다. 이는 `SlideWindowContextChunker` 클래스로 달성됩니다. 정보 추출을 위해 더 낮은 비용을 위해 더 작은 LLM인 llama-3.1-70b-instruct를 사용합니다. 출력 안정성과 재현성을 향상시키기 위해 `temperature = 0.0`으로 설정합니다.

```python
# 합성된 의료 노트 로드
with open("./demo/document/synthesized_note.txt", 'r') as f:
    note_text = f.read()

# 추출기를 위한 LLM 추론 엔진 정의
extractor_llm = OpenAIInferenceEngine(base_url="https://openrouter.ai/api/v1", 
                                      model="meta-llama/llama-3.1-70b-instruct", 
                                      api_key=os.getenv("OPENROUTER_API_KEY"),
                                      config=BasicLLMConfig(temperature=0.0, 
                                                            max_new_tokens=1024))
# 단위 청커 정의. 문장별로 프롬프트합니다.
unit_chunker = SentenceUnitChunker()
# 컨텍스트 청커 정의. 단위에 컨텍스트를 제공합니다.
context_chunker = SlideWindowContextChunker(window_size=2)
# 추출기 정의
extractor = DirectFrameExtractor(inference_engine=extractor_llm, 
                                 unit_chunker=unit_chunker,
                                 context_chunker=context_chunker,
                                 prompt_template=prompt_template)
```

프레임 추출을 실행하려면 `extract_frames` 메서드를 사용하세요. 속성이 있는 개체 목록("프레임")이 반환됩니다. `concurrent=True`로 설정하여 동시 처리를 지원합니다.
```python
# 추출 과정을 스트리밍하려면 concurrent=False, stream=True를 사용하세요:
frames = extractor.extract_frames(note_text, concurrent=False, verbose=True)
# 더 빠른 추출을 위해 concurrent=True를 사용하여 비동기 프롬프팅을 활성화하세요
# frames = extractor.extract_frames(note_text, concurrent=True)

# 추출 확인
for frame in frames:
    print(frame.to_dict())
```

### 데이터 관리 및 시각화
더 나은 관리를 위해 프레임을 문서 객체에 저장할 수 있습니다. 문서는 ```text```와 ```frames```를 보유합니다. ```add_frame()``` 메서드는 유효성 검사를 수행하고 (통과하면) 문서에 프레임을 추가합니다.

```python
from llm_ie.data_types import LLMInformationExtractionDocument

# 문서 정의
doc = LLMInformationExtractionDocument(doc_id="합성된 의료 노트",
                                       text=note_text)
# 문서에 프레임 추가
doc.add_frames(frames, create_id=True)

# 문서를 파일(.llmie)에 저장
doc.save("<파일명>.llmie")
```

추출된 프레임을 시각화하려면 ```viz_serve()``` 메서드를 사용합니다.
```python
doc.viz_serve()
```
Flask 앱이 포트 5000(기본값)에서 시작됩니다.

<div align="left"><img src="docs/readme_img/llm-ie_demo.PNG" width=1000 ></div>

## 🌎웹 애플리케이션
*LLM-IE*에 코드 없이 액세스할 수 있는 드래그 앤 드롭 웹 애플리케이션입니다.

### 설치
이미지는 🐳Docker Hub에서 사용할 수 있습니다. 아래 명령을 사용하여 로컬로 풀하고 실행하세요:
```bash
docker pull daviden1013/llm-ie-web-app:latest
docker run -p 5000:5000 daviden1013/llm-ie-web-app:latest
```

### 기능
프롬프트 에디터 LLM 에이전트와 채팅하기 위한 인터페이스.
![web_app_prompt_editor](docs/readme_img/web_app_prompt_editor.PNG)

스트림 프레임 추출 및 출력 다운로드.
![web_app_frame_extractor](docs/readme_img/web_app_frame_extraction.PNG)

## 🚀배치 처리
LLM-IE는 대용량 텍스트 처리를 위한 고급 배치 처리 기능을 제공합니다.

### Gemini Direct API 배치 처리
웹 애플리케이션은 Google의 Gemini API를 직접 사용하는 특별한 배치 처리 기능을 제공합니다:

#### 주요 기능:
- **다중 API 키 지원**: 여러 API 키를 순환 사용하여 속도 제한 회피
- **청크 기반 처리**: 큰 텍스트를 관리 가능한 청크로 분할
- **실시간 진행 상황**: 스트리밍으로 실시간 처리 상황 확인
- **구성 가능한 매개변수**: 청크 크기, 중복, 배치 크기, 지연 시간 등 조정 가능
- **오류 처리**: 강력한 오류 처리 및 재시도 메커니즘

#### 웹 UI에서 배치 처리 사용법:
1. **배치 처리 활성화**: 프레임 추출 페이지에서 "배치 처리 활성화" 체크박스 선택
2. **배치 유형 선택**: "Gemini Direct" 또는 "LLM-IE 호환" 중 선택
3. **API 키 설정**: 여러 API 키를 한 줄에 하나씩 입력
4. **매개변수 구성**:
   - **청크 크기**: 각 청크의 문자 수 (기본값: 1000)
   - **중복 크기**: 청크 간 중복 문자 수 (기본값: 100)
   - **배치 크기**: 동시 처리할 청크 수 (기본값: 5)
   - **배치 간 지연**: 배치 간 대기 시간 (초) (기본값: 2.0)

#### 예시 구성:
```
API 키들:
AIzaSyC1234567890abcdef...
AIzaSyD0987654321fedcba...

청크 크기: 1500
중복 크기: 150
배치 크기: 3
배치 간 지연: 1.5초
```

### 프로그래밍 방식 배치 처리
Python 코드에서 배치 처리를 사용하려면:

```python
from web_app.app.batch_services import BatchProcessor
from web_app.app.gemini_direct_engine import GeminiDirectBatchProcessor

# LLM-IE 호환 배치 처리
llm_config = {"api_type": "openai_compatible", "model": "gpt-4"}
api_keys = ["key1", "key2", "key3"]
batch_processor = BatchProcessor(llm_config, api_keys)

# Gemini Direct 배치 처리
gemini_config = {
    'gemini_model': 'gemini-2.0-flash',
    'temperature': 0.2,
    'max_tokens': 4096
}
gemini_processor = GeminiDirectBatchProcessor(api_keys, gemini_config)
```

### 성능 최적화
- **API 키 순환**: 속도 제한을 피하기 위해 여러 키 사용
- **청크 크기 조정**: 모델의 컨텍스트 윈도우에 맞게 조정
- **배치 크기 최적화**: 처리 속도와 리소스 사용량 균형
- **지연 시간 설정**: API 제한을 준수하도록 배치 간 지연 조정

## 📘예제
- [LLM 프롬프트 에디터와 대화형 채팅](demo/prompt_template_writing_via_chat.ipynb)
- [LLM 프롬프트 에디터로 프롬프트 템플릿 작성](demo/prompt_template_writing.ipynb)
- [약물, 강도, 빈도에 대한 NER + RE](demo/medication_relation_extraction.ipynb)

## 🔧유용한 스크립트
많은 문서 처리를 위한 템플릿 데이터 파이프라인이 [여기](package/llm-ie/pipelines/)에서 사용할 수 있습니다. 사용 사례에 따라 수정하세요.
- [많은 문서를 순차적으로 처리](package/llm-ie/pipelines/sequential_frame_extraction.py)
- [멀티스레딩으로 많은 문서 처리](package/llm-ie/pipelines/multithread_frame_extraction.py)

## 📐사용자 가이드
자세한 사용자 가이드는 우리의 [문서 페이지](https://daviden1013.github.io/llm-ie/)에서 사용할 수 있습니다.

## 📊벤치마크
생의학 정보 추출 작업에서 프레임 및 관계 추출기를 벤치마킹했습니다. 결과와 실험 코드는 [이 페이지](https://github.com/daviden1013/LLM-IE_Benchmark)에서 사용할 수 있습니다.

## 🎓인용
더 많은 정보와 벤치마크는 우리의 논문을 확인하세요:
```bibtex
@article{hsu2025llm,
  title={LLM-IE: a python package for biomedical generative information extraction with large language models},
  author={Hsu, Enshuo and Roberts, Kirk},
  journal={JAMIA open},
  volume={8},
  number={2},
  pages={ooaf012},
  year={2025},
  publisher={Oxford University Press}
}
```