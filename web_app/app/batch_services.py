import time
import random
import asyncio
import logging
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass
from threading import Lock
from llm_ie.engines import InferenceEngine
from .app_services import create_llm_engine_from_config

logger = logging.getLogger(__name__)

@dataclass
class APIKeyManager:
    """API 키 배치 관리 및 로테이션"""
    api_keys: List[str]
    current_index: int = 0
    usage_count: Dict[str, int] = None
    rate_limits: Dict[str, float] = None  # 각 키별 마지막 사용 시간
    
    def __post_init__(self):
        if self.usage_count is None:
            self.usage_count = {key: 0 for key in self.api_keys}
        if self.rate_limits is None:
            self.rate_limits = {key: 0.0 for key in self.api_keys}
        self._lock = Lock()
    
    def get_next_key(self, min_delay_seconds: float = 1.0) -> str:
        """다음 사용 가능한 API 키 반환 (rate limit 고려)"""
        with self._lock:
            current_time = time.time()
            
            # 가장 오래 전에 사용된 키 찾기
            available_keys = []
            for key in self.api_keys:
                if current_time - self.rate_limits[key] >= min_delay_seconds:
                    available_keys.append(key)
            
            if not available_keys:
                # 모든 키가 rate limit에 걸렸을 경우 대기
                min_wait_time = min(self.rate_limits.values())
                wait_time = min_delay_seconds - (current_time - min_wait_time)
                if wait_time > 0:
                    logger.warning(f"All API keys rate limited. Waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                available_keys = self.api_keys
            
            # Round-robin으로 키 선택
            selected_key = available_keys[self.current_index % len(available_keys)]
            self.current_index = (self.current_index + 1) % len(available_keys)
            
            # 사용 기록 업데이트
            self.usage_count[selected_key] += 1
            self.rate_limits[selected_key] = current_time
            
            logger.info(f"Selected API key: {selected_key[:8]}*** (usage: {self.usage_count[selected_key]})")
            return selected_key

class BatchProcessor:
    """배치 처리를 위한 클래스"""
    
    def __init__(self, base_config: Dict[str, Any], api_keys: List[str]):
        self.base_config = base_config
        self.api_key_manager = APIKeyManager(api_keys)
        self.engines_cache: Dict[str, InferenceEngine] = {}
    
    def get_engine_for_key(self, api_key: str) -> InferenceEngine:
        """API 키별 엔진 캐시 관리"""
        if api_key not in self.engines_cache:
            config = self.base_config.copy()
            
            # API 타입별로 키 필드 이름 매핑
            key_field_mapping = {
                'litellm': 'litellm_api_key',
                'openai': 'openai_api_key',
                'azure_openai': 'azure_openai_api_key',
                'openai_compatible': 'openai_compatible_api_key',
                'huggingface_hub': 'hf_token'
            }
            
            api_type = config.get('api_type', 'litellm')
            key_field = key_field_mapping.get(api_type, 'api_key')
            config[key_field] = api_key
            
            self.engines_cache[api_key] = create_llm_engine_from_config(config)
            logger.info(f"Created engine for API key: {api_key[:8]}***")
        
        return self.engines_cache[api_key]
    
    def process_chunks_batch(self, 
                           text_chunks: List[str], 
                           extractor_config: Dict[str, Any],
                           batch_size: int = 5,
                           delay_between_batches: float = 2.0) -> Iterator[Dict[str, Any]]:
        """청크들을 배치로 나누어 처리"""
        
        total_chunks = len(text_chunks)
        processed_chunks = 0
        
        # 배치 단위로 처리
        for i in range(0, total_chunks, batch_size):
            batch_chunks = text_chunks[i:i + batch_size]
            
            yield {
                "type": "batch_start",
                "data": {
                    "batch_number": i // batch_size + 1,
                    "batch_size": len(batch_chunks),
                    "total_batches": (total_chunks + batch_size - 1) // batch_size,
                    "progress": f"{processed_chunks}/{total_chunks}"
                }
            }
            
            # 배치 내 청크들 병렬 처리
            batch_results = []
            for chunk_idx, chunk in enumerate(batch_chunks):
                try:
                    # 다음 사용 가능한 API 키 선택
                    api_key = self.api_key_manager.get_next_key()
                    engine = self.get_engine_for_key(api_key)
                    
                    # 청크 처리
                    yield {
                        "type": "chunk_start",
                        "data": {
                            "chunk_index": processed_chunks + chunk_idx,
                            "chunk_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                            "api_key_preview": api_key[:8] + "***"
                        }
                    }
                    
                    # 실제 추출 처리 (여기서는 간단한 예시)
                    # 실제로는 extractor를 사용해야 함
                    result = self._process_single_chunk(engine, chunk, extractor_config)
                    
                    batch_results.append({
                        "chunk_index": processed_chunks + chunk_idx,
                        "result": result,
                        "status": "success"
                    })
                    
                    yield {
                        "type": "chunk_complete",
                        "data": {
                            "chunk_index": processed_chunks + chunk_idx,
                            "result": result
                        }
                    }
                    
                    processed_chunks += 1
                    
                except Exception as e:
                    error_result = {
                        "chunk_index": processed_chunks + chunk_idx,
                        "error": str(e),
                        "status": "error"
                    }
                    batch_results.append(error_result)
                    
                    yield {
                        "type": "chunk_error",
                        "data": error_result
                    }
                    
                    processed_chunks += 1
                    logger.error(f"Error processing chunk {processed_chunks + chunk_idx}: {e}")
            
            yield {
                "type": "batch_complete",
                "data": {
                    "batch_number": i // batch_size + 1,
                    "results": batch_results,
                    "progress": f"{processed_chunks}/{total_chunks}"
                }
            }
            
            # 배치 간 대기 (rate limit 방지)
            if i + batch_size < total_chunks:
                logger.info(f"Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)
        
        yield {
            "type": "processing_complete",
            "data": {
                "total_processed": processed_chunks,
                "api_key_usage": dict(self.api_key_manager.usage_count)
            }
        }
    
    def _process_single_chunk(self, engine: InferenceEngine, chunk: str, extractor_config: Dict[str, Any]) -> Dict[str, Any]:
        """단일 청크 처리 - 실제 추출 로직"""
        try:
            from .app_services import get_app_frame_extractor
            
            # AppDirectFrameExtractor 생성
            extractor = get_app_frame_extractor(engine, extractor_config)
            
            # 청크에서 프레임 추출
            extraction_results = []
            stream_generator = extractor.stream(
                text_content=chunk,
                document_key=None
            )
            
            # 스트림에서 결과 수집
            while True:
                try:
                    event = next(stream_generator)
                    # 진행 상황 로그 (선택적)
                    if event.get('type') in ['info', 'debug']:
                        logger.debug(f"Chunk extraction event: {event.get('type')}")
                except StopIteration as e:
                    extraction_results = e.value if e.value else []
                    break
            
            # 후처리
            final_frames = extractor.post_process_frames(
                extraction_results=extraction_results,
                case_sensitive=extractor_config.get('case_sensitive', False),
                fuzzy_match=extractor_config.get('fuzzy_match', True),
                allow_overlap_entities=extractor_config.get('allow_overlap_entities', False),
                fuzzy_buffer_size=float(extractor_config.get('fuzzy_buffer_size', 0.2)),
                fuzzy_score_cutoff=float(extractor_config.get('fuzzy_score_cutoff', 0.8))
            )
            
            return {
                "chunk_text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "chunk_length": len(chunk),
                "raw_extraction_count": len(extraction_results),
                "final_frame_count": len(final_frames),
                "extracted_frames": [frame.to_dict() for frame in final_frames],
                "processing_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {str(e)}")
            raise Exception(f"Chunk processing failed: {str(e)}")


def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[tuple[str, int]]:
    """텍스트를 지정된 크기로 청크 분할하고 각 청크의 시작 위치 반환"""
    if len(text) <= chunk_size:
        return [(text, 0)]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # 현재 청크의 끝 위치 계산
        end = min(start + chunk_size, len(text))
        
        # 마지막 청크가 아니고 문장 경계에서 자를 수 있다면 조정
        if end < len(text):
            # 뒤에서부터 마침표, 줄바꿈, 공백 찾기
            sentence_boundaries = [
                text.rfind('.', start, end),
                text.rfind('!', start, end),
                text.rfind('?', start, end),
                text.rfind('\n', start, end),
                text.rfind(' ', start, end)
            ]
            
            best_boundary = max([b for b in sentence_boundaries if b > start + chunk_size // 2], default=-1)
            if best_boundary > start:
                end = best_boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start))
            logger.info(f"Created chunk {len(chunks)}: length={len(chunk)}, start={start}, end={end}")
        
        # 다음 시작점 계산
        if end >= len(text):
            break  # 텍스트 끝에 도달
        
        # 겹침을 고려하여 다음 시작점 설정
        next_start = end - overlap
        
        # 진행이 안 되는 경우 방지 (최소한 chunk_size//4 만큼은 진행)
        min_progress = start + max(chunk_size // 4, 50)
        if next_start <= start:
            next_start = min_progress
        
        start = next_start
        
        # 안전장치: 현재 청크와 동일한 위치에서 시작하지 않도록
        if start >= end:
            start = end
    
    logger.info(f"Split text into {len(chunks)} chunks (total length: {len(text)})")
    return chunks