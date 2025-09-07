import json
import time
import logging
import requests
from typing import Dict, Any, Iterator, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GeminiDirectConfig:
    """Gemini 직접 호출을 위한 설정"""
    api_key: str
    model: str = "gemini-2.0-flash"
    temperature: float = 0.2
    max_output_tokens: int = 4096
    top_p: float = 0.95
    top_k: int = 40

class GeminiDirectEngine:
    """Gemini API 직접 호출 엔진"""
    
    def __init__(self, config: GeminiDirectConfig):
        self.config = config
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
    def _build_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """메시지를 Gemini API 형식으로 변환"""
        contents = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                # System 메시지는 첫 번째 user 메시지에 포함
                if contents and contents[0]['role'] == 'user':
                    contents[0]['parts'][0]['text'] = f"{content}\n\n{contents[0]['parts'][0]['text']}"
                else:
                    contents.insert(0, {
                        'role': 'user',
                        'parts': [{'text': content}]
                    })
            elif role == 'user':
                contents.append({
                    'role': 'user', 
                    'parts': [{'text': content}]
                })
            elif role == 'assistant':
                contents.append({
                    'role': 'model',
                    'parts': [{'text': content}]
                })
        
        return {
            'contents': contents,
            'generationConfig': {
                'temperature': self.config.temperature,
                'maxOutputTokens': self.config.max_output_tokens,
                'topP': self.config.top_p,
                'topK': self.config.top_k,
            }
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], stream: bool = False) -> Iterator[Dict[str, Any]]:
        """채팅 완료 요청"""
        payload = self._build_payload(messages)
        
        if stream:
            return self._stream_request(payload)
        else:
            return self._single_request(payload)
    
    def _single_request(self, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """단일 요청 (non-streaming)"""
        url = f"{self.base_url}/models/{self.config.model}:generateContent"
        params = {'key': self.config.api_key}
        
        try:
            response = requests.post(
                url, 
                params=params,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            
            if response.status_code == 429:
                # Rate limit 처리
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self._single_request(payload)  # 재시도
            
            response.raise_for_status()
            result = response.json()
            
            if 'candidates' in result and result['candidates']:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    text = candidate['content']['parts'][0].get('text', '')
                    yield {
                        'type': 'content',
                        'text': text,
                        'done': True
                    }
                else:
                    logger.error(f"Unexpected response structure: {result}")
                    yield {
                        'type': 'error',
                        'error': 'Unexpected response structure'
                    }
            else:
                logger.error(f"No candidates in response: {result}")
                yield {
                    'type': 'error', 
                    'error': 'No content generated'
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request failed: {e}")
            yield {
                'type': 'error',
                'error': str(e)
            }
    
    def _stream_request(self, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """스트리밍 요청"""
        url = f"{self.base_url}/models/{self.config.model}:streamGenerateContent"
        params = {'key': self.config.api_key, 'alt': 'sse'}
        
        try:
            response = requests.post(
                url,
                params=params,
                json=payload,
                headers={'Content-Type': 'application/json'},
                stream=True,
                timeout=60
            )
            
            if response.status_code == 429:
                # Rate limit 처리
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                yield from self._stream_request(payload)  # 재시도
                return
            
            response.raise_for_status()
            
            buffer = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                    
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    
                    if data_str.strip() == '[DONE]':
                        yield {'type': 'done'}
                        break
                    
                    try:
                        data = json.loads(data_str)
                        
                        if 'candidates' in data and data['candidates']:
                            candidate = data['candidates'][0]
                            if 'content' in candidate and 'parts' in candidate['content']:
                                text = candidate['content']['parts'][0].get('text', '')
                                if text:
                                    buffer += text
                                    yield {
                                        'type': 'content',
                                        'text': text,
                                        'done': False
                                    }
                                    
                        # 완료 확인
                        if 'candidates' in data:
                            for candidate in data['candidates']:
                                if candidate.get('finishReason'):
                                    yield {
                                        'type': 'content',
                                        'text': '',
                                        'done': True,
                                        'total_text': buffer
                                    }
                                    return
                                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse streaming data: {e}, data: {data_str}")
                        continue
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini streaming request failed: {e}")
            yield {
                'type': 'error',
                'error': str(e)
            }

class GeminiDirectBatchProcessor:
    """Gemini 직접 호출을 사용한 배치 프로세서"""
    
    def __init__(self, api_keys: List[str], base_config: Dict[str, Any]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.key_usage_count = {key: 0 for key in api_keys}
        self.last_request_time = {key: 0.0 for key in api_keys}
        
        # Gemini 설정
        self.model = base_config.get('gemini_model', 'gemini-2.0-flash')
        self.temperature = float(base_config.get('temperature', 0.2))
        self.max_tokens = int(base_config.get('max_tokens', 4096))
        
    def get_next_engine(self, min_delay: float = 1.0) -> GeminiDirectEngine:
        """다음 사용 가능한 API 키로 엔진 생성"""
        current_time = time.time()
        
        # Rate limit 회피를 위한 키 선택
        available_keys = []
        for i, key in enumerate(self.api_keys):
            if current_time - self.last_request_time[key] >= min_delay:
                available_keys.append((i, key))
        
        if not available_keys:
            # 모든 키가 대기 중이면 가장 오래된 키 선택 후 대기
            oldest_key = min(self.api_keys, key=lambda k: self.last_request_time[k])
            wait_time = min_delay - (current_time - self.last_request_time[oldest_key])
            if wait_time > 0:
                logger.info(f"All keys rate limited. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            selected_key = oldest_key
        else:
            # Round-robin으로 사용 가능한 키 선택
            key_index = self.current_key_index % len(available_keys)
            _, selected_key = available_keys[key_index]
            self.current_key_index = (self.current_key_index + 1) % len(available_keys)
        
        # 사용 기록 업데이트
        self.key_usage_count[selected_key] += 1
        self.last_request_time[selected_key] = time.time()
        
        logger.info(f"Using Gemini API key: {selected_key[:8]}*** (usage: {self.key_usage_count[selected_key]})")
        
        config = GeminiDirectConfig(
            api_key=selected_key,
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )
        
        return GeminiDirectEngine(config)
    
    def process_chunk(self, chunk: str, prompt_template: str, frame_id_offset: int = 0, chunk_start_position: int = 0) -> Dict[str, Any]:
        """단일 청크 처리 및 JSON 결과 파싱"""
        engine = self.get_next_engine()
        
        # 프롬프트 템플릿에서 {{text}} 대체
        full_prompt = prompt_template.replace('{{text}}', chunk)
        
        messages = [
            {'role': 'user', 'content': full_prompt}
        ]
        
        try:
            # 단일 요청으로 처리
            response_generator = engine.chat_completion(messages, stream=False)
            
            for response in response_generator:
                if response.get('type') == 'content' and response.get('done'):
                    raw_text = response.get('text', '').strip()
                    
                    # JSON 응답 파싱 시도 (소스 텍스트 및 위치 정보 전달)
                    extracted_frames = self._parse_extraction_response(raw_text, chunk, frame_id_offset, chunk_start_position)
                    
                    return {
                        'success': True,
                        'raw_content': raw_text,
                        'extracted_frames': extracted_frames,
                        'chunk_text': chunk,
                        'chunk_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
                    }
                elif response.get('type') == 'error':
                    return {
                        'success': False,
                        'error': response.get('error', 'Unknown error'),
                        'chunk_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
                    }
            
            return {
                'success': False,
                'error': 'No response received',
                'chunk_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
            }
            
        except Exception as e:
            logger.error(f"Chunk processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunk_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
            }
    
    def _parse_extraction_response(self, raw_text: str, source_text: str = "", frame_id_offset: int = 0, chunk_start_position: int = 0) -> List[Dict[str, Any]]:
        """LLM 응답에서 JSON 구조 추출"""
        try:
            # JSON 블록 찾기
            import re
            
            # ```json과 ``` 사이의 내용 찾기
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', raw_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # [] 로 둘러싸인 첫 번째 JSON 배열 찾기
                json_match = re.search(r'(\[[\s\S]*?\])', raw_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # 전체 텍스트가 JSON인 경우
                    json_str = raw_text.strip()
            
            # JSON 파싱
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    return self._convert_to_frames(parsed, source_text, frame_id_offset, chunk_start_position)
                else:
                    logger.warning(f"Expected list, got {type(parsed)}: {parsed}")
                    return []
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed: {e}, raw: {json_str[:200]}...")
                return []
                
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return []
    
    def _convert_to_frames(self, json_data: List[Dict], source_text: str = "", frame_id_offset: int = 0, chunk_start_position: int = 0) -> List[Dict[str, Any]]:
        """JSON 데이터를 LLM-IE Frame 형식으로 변환"""
        frames = []
        
        for i, item in enumerate(json_data):
            if not isinstance(item, dict):
                continue
            
            entity_text = item.get('entity_text', '')
            if not entity_text:
                continue
            
            # 텍스트에서 엔티티 위치 찾기 (전체 문서 기준)
            start_pos, end_pos = self._find_entity_position(entity_text, source_text, chunk_start_position)
            
            # LLM-IE 호환 frame 구조 생성 - 고유한 frame_id 사용
            frame = {
                'frame_id': str(frame_id_offset + i),  # 오프셋을 추가하여 고유성 보장
                'start': start_pos,
                'end': end_pos, 
                'entity_text': entity_text,
                'attr': item.get('attr', {}),
                'confidence': item.get('confidence', 1.0)
            }
            
            frames.append(frame)
        
        logger.info(f"Converted {len(frames)} frames from JSON response with position info")
        return frames
    
    def _find_entity_position(self, entity_text: str, source_text: str, chunk_start_position: int = 0) -> tuple[int, int]:
        """소스 텍스트에서 엔티티의 위치를 찾고 전체 문서 기준으로 변환"""
        try:
            # 청크 내에서 엔티티 위치 찾기
            start_pos = source_text.find(entity_text)
            if start_pos != -1:
                end_pos = start_pos + len(entity_text)
                # 전체 문서 기준 위치로 변환
                global_start_pos = chunk_start_position + start_pos
                global_end_pos = chunk_start_position + end_pos
                logger.debug(f"Entity '{entity_text}' found at chunk position ({start_pos}, {end_pos}), global position ({global_start_pos}, {global_end_pos})")
                return global_start_pos, global_end_pos
            else:
                # 정확히 일치하는 부분을 찾지 못한 경우, 유사한 부분 찾기
                import re
                # 공백이나 특수문자 차이 무시하고 찾기
                normalized_entity = re.sub(r'\s+', r'\\s+', re.escape(entity_text))
                match = re.search(normalized_entity, source_text, re.IGNORECASE)
                if match:
                    # 전체 문서 기준 위치로 변환
                    global_start_pos = chunk_start_position + match.start()
                    global_end_pos = chunk_start_position + match.end()
                    logger.debug(f"Entity '{entity_text}' found (fuzzy) at chunk position ({match.start()}, {match.end()}), global position ({global_start_pos}, {global_end_pos})")
                    return global_start_pos, global_end_pos
                else:
                    # 찾지 못한 경우 기본값
                    logger.warning(f"Could not find position for entity: {entity_text} in chunk starting at {chunk_start_position}")
                    return -1, -1
        except Exception as e:
            logger.error(f"Error finding entity position: {e}")
            return -1, -1