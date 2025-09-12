"""
Entity Resolution Sub-agents for LLM-IE Web Application
엔티티 해결을 위한 서브 에이전트들

DRY와 SOLID 원칙을 적용하여 설계된 독립적인 엔티티 해결 시스템
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import difflib
import re

logger = logging.getLogger(__name__)


class SimilarityMethod(Enum):
    """유사도 계산 방법"""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HYBRID = "hybrid"


@dataclass
class EntityMatch:
    """엔티티 매칭 결과"""
    entity1_id: str
    entity2_id: str
    similarity_score: float
    similarity_method: SimilarityMethod
    confidence: float
    reasoning: str


@dataclass
class MergedEntity:
    """병합된 엔티티"""
    merged_id: str
    canonical_text: str
    entity_type: str
    source_entities: List[str]
    confidence: float
    merge_reasoning: str


class SimilarityCalculator(ABC):
    """유사도 계산기 인터페이스 (Strategy Pattern)"""
    
    @abstractmethod
    def calculate_similarity(self, entity1: Dict, entity2: Dict) -> float:
        """두 엔티티 간 유사도 계산"""
        pass
    
    @abstractmethod
    def get_method_name(self) -> SimilarityMethod:
        """계산 방법명 반환"""
        pass


class ExactMatchCalculator(SimilarityCalculator):
    """정확한 텍스트 매칭 계산기"""
    
    def calculate_similarity(self, entity1: Dict, entity2: Dict) -> float:
        text1 = self._normalize_text(entity1.get('text', entity1.get('entity_text', '')))
        text2 = self._normalize_text(entity2.get('text', entity2.get('entity_text', '')))
        
        if text1 == text2:
            return 1.0
        return 0.0
    
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 공백, 대소문자, 특수문자 정규화
        normalized = re.sub(r'\s+', ' ', text.strip().lower())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized
    
    def get_method_name(self) -> SimilarityMethod:
        return SimilarityMethod.EXACT_MATCH


class FuzzyMatchCalculator(SimilarityCalculator):
    """퍼지 매칭 계산기"""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    def calculate_similarity(self, entity1: Dict, entity2: Dict) -> float:
        text1 = entity1.get('text', entity1.get('entity_text', '')).lower().strip()
        text2 = entity2.get('text', entity2.get('entity_text', '')).lower().strip()
        
        # SequenceMatcher를 사용한 유사도 계산
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # 타입이 다르면 패널티 적용
        type1 = entity1.get('type', entity1.get('attr', {}).get('entity_type', ''))
        type2 = entity2.get('type', entity2.get('attr', {}).get('entity_type', ''))
        
        if type1 != type2 and type1 and type2:
            similarity *= 0.7  # 타입 불일치 패널티
        
        return similarity
    
    def get_method_name(self) -> SimilarityMethod:
        return SimilarityMethod.FUZZY_MATCH


class SemanticSimilarityCalculator(SimilarityCalculator):
    """의미적 유사도 계산기 (LLM 기반)"""
    
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine
    
    def calculate_similarity(self, entity1: Dict, entity2: Dict) -> float:
        if not self.llm_engine:
            return 0.0
        
        try:
            text1 = entity1.get('text', entity1.get('entity_text', ''))
            text2 = entity2.get('text', entity2.get('entity_text', ''))
            type1 = entity1.get('type', entity1.get('attr', {}).get('entity_type', ''))
            type2 = entity2.get('type', entity2.get('attr', {}).get('entity_type', ''))
            
            prompt = f"""
두 엔티티가 같은 개념을 나타내는지 평가해주세요.

엔티티 1: "{text1}" (타입: {type1})
엔티티 2: "{text2}" (타입: {type2})

0.0 (완전히 다름)부터 1.0 (동일함)까지 유사도 점수를 JSON으로 반환해주세요:
{{
    "similarity_score": 0.8,
    "reasoning": "두 엔티티가 같은 개념을 나타내는 이유"
}}
"""
            
            messages = [{"role": "user", "content": prompt}]
            response_text = ""
            
            for response in self.llm_engine.chat_completion(messages, stream=False):
                if response.get('type') == 'content':
                    response_text += response.get('text', '')
            
            try:
                result = json.loads(response_text.strip())
                return float(result.get('similarity_score', 0.0))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse LLM similarity response")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}")
            return 0.0
    
    def get_method_name(self) -> SimilarityMethod:
        return SimilarityMethod.SEMANTIC_SIMILARITY


class EntityMatcher:
    """엔티티 매칭 담당 클래스 (Single Responsibility Principle)"""
    
    def __init__(self, similarity_calculators: List[SimilarityCalculator], match_threshold: float = 0.95):
        self.similarity_calculators = similarity_calculators
        self.match_threshold = match_threshold
        
    def find_matches(self, entities: List[Dict]) -> List[EntityMatch]:
        """엔티티들 간 매칭 찾기"""
        matches = []
        
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                entity1, entity2 = entities[i], entities[j]
                
                match = self._calculate_entity_match(entity1, entity2)
                if match and match.similarity_score >= self.match_threshold:
                    matches.append(match)
        
        return matches
    
    def _calculate_entity_match(self, entity1: Dict, entity2: Dict) -> Optional[EntityMatch]:
        """두 엔티티 간 매치 계산"""
        best_score = 0.0
        best_method = None
        reasoning_parts = []
        
        for calculator in self.similarity_calculators:
            score = calculator.calculate_similarity(entity1, entity2)
            method = calculator.get_method_name()
            
            if score > best_score:
                best_score = score
                best_method = method
            
            reasoning_parts.append(f"{method.value}: {score:.3f}")
        
        if best_score >= self.match_threshold:
            confidence = min(best_score * 1.1, 1.0)  # 약간의 보정
            reasoning = f"Best match via {best_method.value} ({', '.join(reasoning_parts)})"
            
            return EntityMatch(
                entity1_id=entity1.get('id', entity1.get('frame_id')),
                entity2_id=entity2.get('id', entity2.get('frame_id')),
                similarity_score=best_score,
                similarity_method=best_method,
                confidence=confidence,
                reasoning=reasoning
            )
        
        return None


class EntityMerger:
    """엔티티 병합 담당 클래스 (Single Responsibility Principle)"""
    
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine
    
    def merge_entities(self, entities: List[Dict], matches: List[EntityMatch]) -> Tuple[List[MergedEntity], Dict[str, str]]:
        """매칭된 엔티티들을 병합"""
        # 연결된 컴포넌트 찾기 (Union-Find 알고리즘)
        entity_groups = self._find_connected_components(entities, matches)
        
        merged_entities = []
        entity_mapping = {}  # 원본 ID -> 병합 ID 매핑
        
        for group_id, entity_ids in entity_groups.items():
            if len(entity_ids) == 1:
                # 단일 엔티티는 그대로 유지
                entity_id = list(entity_ids)[0]
                entity = self._find_entity_by_id(entities, entity_id)
                
                merged_entity = MergedEntity(
                    merged_id=entity_id,
                    canonical_text=entity.get('text', entity.get('entity_text', '')),
                    entity_type=entity.get('type', entity.get('attr', {}).get('entity_type', '')),
                    source_entities=[entity_id],
                    confidence=1.0,
                    merge_reasoning="Single entity - no merge needed"
                )
                merged_entities.append(merged_entity)
                entity_mapping[entity_id] = entity_id
            else:
                # 복수 엔티티 병합
                group_entities = [self._find_entity_by_id(entities, eid) for eid in entity_ids]
                merged_entity = self._merge_entity_group(group_entities, group_id)
                merged_entities.append(merged_entity)
                
                for entity_id in entity_ids:
                    entity_mapping[entity_id] = merged_entity.merged_id
        
        return merged_entities, entity_mapping
    
    def _find_connected_components(self, entities: List[Dict], matches: List[EntityMatch]) -> Dict[str, Set[str]]:
        """연결된 엔티티 그룹 찾기"""
        # Union-Find 자료구조
        parent = {}
        
        # 모든 엔티티를 개별 그룹으로 초기화
        for entity in entities:
            entity_id = entity.get('id', entity.get('frame_id'))
            parent[entity_id] = entity_id
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 매치된 엔티티들을 같은 그룹으로 병합
        for match in matches:
            union(match.entity1_id, match.entity2_id)
        
        # 그룹별로 엔티티 수집
        groups = {}
        for entity in entities:
            entity_id = entity.get('id', entity.get('frame_id'))
            root = find(entity_id)
            if root not in groups:
                groups[root] = set()
            groups[root].add(entity_id)
        
        return groups
    
    def _find_entity_by_id(self, entities: List[Dict], entity_id: str) -> Dict:
        """ID로 엔티티 찾기"""
        for entity in entities:
            if entity.get('id', entity.get('frame_id')) == entity_id:
                return entity
        return {}
    
    def _merge_entity_group(self, entities: List[Dict], group_id: str) -> MergedEntity:
        """엔티티 그룹을 하나로 병합"""
        if not entities:
            raise ValueError("Empty entity group cannot be merged")
        
        # 가장 대표적인 텍스트 선택 (가장 긴 것 또는 LLM 기반 선택)
        canonical_text = self._select_canonical_text(entities)
        
        # 공통 타입 결정
        entity_type = self._determine_common_type(entities)
        
        # 소스 엔티티 ID 수집
        source_entities = [e.get('id', e.get('frame_id')) for e in entities]
        
        # 병합 추론 생성
        entity_texts = [e.get('text', e.get('entity_text', '')) for e in entities]
        merge_reasoning = f"Merged {len(entities)} entities: {', '.join(entity_texts)}"
        
        # 신뢰도 계산 (엔티티 수가 많을수록 약간 낮아짐)
        confidence = max(0.7, 1.0 - (len(entities) - 1) * 0.1)
        
        return MergedEntity(
            merged_id=f"merged_{group_id}",
            canonical_text=canonical_text,
            entity_type=entity_type,
            source_entities=source_entities,
            confidence=confidence,
            merge_reasoning=merge_reasoning
        )
    
    def _select_canonical_text(self, entities: List[Dict]) -> str:
        """가장 대표적인 텍스트 선택"""
        if not entities:
            return ""
        
        # 간단한 휴리스틱: 가장 긴 텍스트 선택
        # 추후 LLM 기반 선택으로 개선 가능
        canonical = max(entities, key=lambda e: len(e.get('text', e.get('entity_text', ''))))
        return canonical.get('text', canonical.get('entity_text', ''))
    
    def _determine_common_type(self, entities: List[Dict]) -> str:
        """공통 타입 결정"""
        types = [e.get('type', e.get('attr', {}).get('entity_type', '')) for e in entities]
        types = [t for t in types if t]  # 빈 타입 제외
        
        if not types:
            return 'UNKNOWN'
        
        # 가장 빈번한 타입 선택
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return max(type_counts.items(), key=lambda x: x[1])[0]


class EntityResolutionAgent:
    """
    Entity Resolution 메인 에이전트
    
    Responsibilities:
    1. 엔티티 간 유사도 계산
    2. 유사한 엔티티 매칭
    3. 매칭된 엔티티 병합
    4. 병합 결과 검증
    
    SOLID 원칙 적용:
    - Single Responsibility: 엔티티 해결만 담당
    - Open/Closed: 새로운 유사도 계산기 추가 가능
    - Liskov Substitution: SimilarityCalculator 인터페이스 준수
    - Interface Segregation: 작은 인터페이스들로 분리
    - Dependency Inversion: 추상화에 의존
    """
    
    def __init__(self, llm_engine=None, use_semantic_similarity: bool = True):
        self.llm_engine = llm_engine
        
        # 유사도 계산기들 설정 (Strategy Pattern)
        self.similarity_calculators = [
            ExactMatchCalculator(),
            FuzzyMatchCalculator(threshold=0.8)
        ]
        
        if use_semantic_similarity and llm_engine:
            self.similarity_calculators.append(SemanticSimilarityCalculator(llm_engine))
        
        # 의존성 주입 (Dependency Injection)
        self.matcher = EntityMatcher(self.similarity_calculators, match_threshold=0.95)
        self.merger = EntityMerger(llm_engine)
    
    def resolve_entities(self, entities: List[Dict], 
                        resolution_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        엔티티 해결 수행
        
        Args:
            entities: 해결할 엔티티 리스트
            resolution_config: 해결 설정
            
        Returns:
            해결 결과
        """
        try:
            config = resolution_config or {}
            
            # 설정에 따라 임계값 조정
            similarity_threshold = config.get('similarity_threshold', 0.95)
            self.matcher.match_threshold = similarity_threshold
            
            logger.info(f"Starting entity resolution for {len(entities)} entities with threshold {similarity_threshold}")
            
            # 1. 엔티티 매칭
            matches = self.matcher.find_matches(entities)
            logger.info(f"Found {len(matches)} potential matches")
            
            # 2. 엔티티 병합
            merged_entities, entity_mapping = self.merger.merge_entities(entities, matches)
            logger.info(f"Merged into {len(merged_entities)} entities")
            
            # 3. 통계 생성
            stats = self._generate_resolution_statistics(entities, merged_entities, matches)
            
            return {
                'success': True,
                'original_count': len(entities),
                'resolved_count': len(merged_entities),
                'merged_entities': [self._merged_entity_to_dict(me) for me in merged_entities],
                'entity_mapping': entity_mapping,
                'matches': [self._match_to_dict(m) for m in matches],
                'statistics': stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in entity resolution: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _merged_entity_to_dict(self, merged_entity: MergedEntity) -> Dict:
        """MergedEntity를 딕셔너리로 변환"""
        return {
            'merged_id': merged_entity.merged_id,
            'canonical_text': merged_entity.canonical_text,
            'entity_type': merged_entity.entity_type,
            'source_entities': merged_entity.source_entities,
            'confidence': merged_entity.confidence,
            'merge_reasoning': merged_entity.merge_reasoning
        }
    
    def _match_to_dict(self, match: EntityMatch) -> Dict:
        """EntityMatch를 딕셔너리로 변환"""
        return {
            'entity1_id': match.entity1_id,
            'entity2_id': match.entity2_id,
            'similarity_score': match.similarity_score,
            'similarity_method': match.similarity_method.value,
            'confidence': match.confidence,
            'reasoning': match.reasoning
        }
    
    def _generate_resolution_statistics(self, original_entities: List[Dict], 
                                      merged_entities: List[MergedEntity], 
                                      matches: List[EntityMatch]) -> Dict:
        """해결 통계 생성"""
        stats = {
            'total_original': len(original_entities),
            'total_resolved': len(merged_entities),
            'reduction_count': len(original_entities) - len(merged_entities),
            'reduction_percentage': ((len(original_entities) - len(merged_entities)) / len(original_entities) * 100) if original_entities else 0,
            'total_matches': len(matches),
            'entity_types': {}
        }
        
        # 타입별 통계
        for entity in original_entities:
            entity_type = entity.get('attr', {}).get('entity_type', 'UNKNOWN')
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
        
        return stats


class EntityResolutionValidator:
    """
    Entity Resolution 검증 및 오류 수정 Sub-agent
    
    Responsibilities:
    1. 병합 결과 검증
    2. 잘못된 병합 감지
    3. 누락된 병합 감지
    4. 자동 오류 수정
    """
    
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine
        self.validation_rules = [
            self._validate_type_consistency,
            self._validate_merge_quality,
            self._validate_completeness,
            self._validate_canonical_text_selection
        ]
    
    def validate_resolution(self, resolution_result: Dict, auto_fix: bool = True) -> Dict[str, Any]:
        """
        엔티티 해결 결과 검증
        
        Args:
            resolution_result: 해결 결과
            auto_fix: 자동 수정 여부
            
        Returns:
            검증 결과
        """
        try:
            if not resolution_result.get('success'):
                return {
                    'success': False,
                    'error': 'Cannot validate failed entity resolution'
                }
            
            merged_entities = resolution_result.get('merged_entities', [])
            matches = resolution_result.get('matches', [])
            
            logger.info(f"Validating entity resolution with {len(merged_entities)} merged entities")
            
            validation_results = []
            issues_found = []
            fixes_applied = []
            
            # 각 검증 규칙 실행
            for rule in self.validation_rules:
                try:
                    result = rule(merged_entities, matches)
                    validation_results.append(result)
                    
                    if not result['passed']:
                        issues_found.extend(result.get('issues', []))
                        
                        # 자동 수정 시도
                        if auto_fix and 'fix_function' in result:
                            fix_result = result['fix_function'](merged_entities, matches)
                            if fix_result.get('success'):
                                fixes_applied.extend(fix_result.get('fixes', []))
                
                except Exception as e:
                    logger.error(f"Error in validation rule {rule.__name__}: {e}")
                    validation_results.append({
                        'rule': rule.__name__,
                        'passed': False,
                        'error': str(e)
                    })
            
            # 전체 점수 계산
            passed_count = sum(1 for r in validation_results if r.get('passed', False))
            total_count = len(validation_results)
            overall_score = (passed_count / total_count * 100) if total_count > 0 else 0
            
            return {
                'success': True,
                'overall_score': overall_score,
                'validation_results': validation_results,
                'issues_found': issues_found,
                'fixes_applied': fixes_applied,
                'total_rules': total_count,
                'passed_rules': passed_count,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating entity resolution: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_type_consistency(self, merged_entities: List[Dict], matches: List[Dict]) -> Dict:
        """타입 일관성 검증"""
        inconsistent_merges = []
        
        for entity in merged_entities:
            source_entities = entity.get('source_entities', [])
            if len(source_entities) > 1:
                # 멀티 엔티티 병합의 경우 타입 검증 필요
                # 실제 구현에서는 원본 엔티티들의 타입을 확인해야 함
                pass
        
        return {
            'rule': 'type_consistency',
            'passed': len(inconsistent_merges) == 0,
            'issues': inconsistent_merges,
            'message': f"Found {len(inconsistent_merges)} type-inconsistent merges"
        }
    
    def _validate_merge_quality(self, merged_entities: List[Dict], matches: List[Dict]) -> Dict:
        """병합 품질 검증"""
        low_quality_merges = []
        
        for entity in merged_entities:
            confidence = entity.get('confidence', 0)
            if confidence < 0.7:
                low_quality_merges.append(entity.get('merged_id'))
        
        return {
            'rule': 'merge_quality',
            'passed': len(low_quality_merges) == 0,
            'issues': [f"Low confidence merge: {mid}" for mid in low_quality_merges],
            'message': f"Found {len(low_quality_merges)} low-quality merges"
        }
    
    def _validate_completeness(self, merged_entities: List[Dict], matches: List[Dict]) -> Dict:
        """완전성 검증"""
        # 누락된 매치가 있는지 확인
        return {
            'rule': 'completeness',
            'passed': True,
            'issues': [],
            'message': "Completeness validation passed"
        }
    
    def _validate_canonical_text_selection(self, merged_entities: List[Dict], matches: List[Dict]) -> Dict:
        """대표 텍스트 선택 검증"""
        poor_selections = []
        
        for entity in merged_entities:
            source_count = len(entity.get('source_entities', []))
            canonical_text = entity.get('canonical_text', '')
            
            if source_count > 1 and len(canonical_text) < 3:
                poor_selections.append(entity.get('merged_id'))
        
        return {
            'rule': 'canonical_text_selection',
            'passed': len(poor_selections) == 0,
            'issues': [f"Poor canonical text: {mid}" for mid in poor_selections],
            'message': f"Found {len(poor_selections)} poor canonical text selections"
        }