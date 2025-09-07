"""
Relation Inference Sub-agents for Knowledge Graph Generation
관계 추론을 위한 서브 에이전트들
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import re


logger = logging.getLogger(__name__)


class RelationPromptAgent:
    """
    관계 추론 프롬프트 생성을 위한 Sub-agent
    컨텍스트와 엔티티 정보를 바탕으로 최적의 관계 추론 프롬프트를 생성
    """
    
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine
        
        # 도메인별 관계 템플릿
        self.domain_templates = {
            "research_paper": {
                "name": "연구 논문",
                "description": "학술 논문의 엔티티 간 관계",
                "common_relations": [
                    "STUDIES", "ANALYZES", "COMPARES", "USES", "MEASURES", 
                    "EVALUATES", "PROPOSES", "IMPLEMENTS", "CITES", "EXTENDS"
                ],
                "relation_patterns": {
                    "MODEL-MODEL": ["COMPARES_WITH", "SIMILAR_TO", "BETTER_THAN"],
                    "RESEARCH-MODEL": ["STUDIES", "ANALYZES", "EVALUATES"],
                    "RESEARCH-SURVEY": ["USES", "IMPLEMENTS", "APPLIES"],
                    "MODEL-TECHNOLOGY": ["IS_TYPE_OF", "IMPLEMENTS", "USES"],
                    "DOCUMENT-RESEARCH": ["DESCRIBES", "PRESENTS", "REPORTS"]
                }
            },
            "technical_document": {
                "name": "기술 문서",
                "description": "기술 문서의 컴포넌트 간 관계",
                "common_relations": [
                    "IMPLEMENTS", "USES", "DEPENDS_ON", "PART_OF", "CONTAINS",
                    "CONNECTS_TO", "CONFIGURES", "MANAGES", "PROCESSES"
                ],
                "relation_patterns": {
                    "COMPONENT-COMPONENT": ["CONNECTS_TO", "DEPENDS_ON", "PART_OF"],
                    "SYSTEM-COMPONENT": ["CONTAINS", "MANAGES", "USES"],
                    "PROCESS-DATA": ["PROCESSES", "TRANSFORMS", "ANALYZES"],
                    "API-SERVICE": ["PROVIDES", "IMPLEMENTS", "EXPOSES"]
                }
            },
            "business_document": {
                "name": "비즈니스 문서",
                "description": "비즈니스 프로세스와 조직 간 관계",
                "common_relations": [
                    "MANAGES", "REPORTS_TO", "COLLABORATES_WITH", "RESPONSIBLE_FOR",
                    "APPROVES", "MONITORS", "SUPPORTS", "DELIVERS", "CONTRACTS_WITH"
                ],
                "relation_patterns": {
                    "PERSON-PERSON": ["REPORTS_TO", "COLLABORATES_WITH", "MANAGES"],
                    "ORGANIZATION-PERSON": ["EMPLOYS", "CONTRACTS_WITH", "PARTNERS_WITH"],
                    "PROCESS-ORGANIZATION": ["MANAGED_BY", "EXECUTED_BY", "OWNED_BY"],
                    "PRODUCT-CUSTOMER": ["USED_BY", "PURCHASED_BY", "REQUESTED_BY"]
                }
            },
            "general": {
                "name": "일반",
                "description": "일반적인 텍스트의 엔티티 간 관계",
                "common_relations": [
                    "RELATED_TO", "PART_OF", "CONTAINS", "DESCRIBES", "MENTIONS",
                    "REFERS_TO", "ASSOCIATED_WITH", "LOCATED_IN", "OCCURS_IN"
                ],
                "relation_patterns": {
                    "ENTITY-ENTITY": ["RELATED_TO", "ASSOCIATED_WITH", "CONNECTED_TO"],
                    "DOCUMENT-ENTITY": ["MENTIONS", "DESCRIBES", "DISCUSSES"],
                    "LOCATION-ENTITY": ["CONTAINS", "LOCATED_IN", "HOSTS"],
                    "TIME-EVENT": ["OCCURS_IN", "DURING", "BEFORE", "AFTER"]
                }
            }
        }
    
    def generate_relation_prompt(self, 
                                text: str, 
                                entities: List[Dict], 
                                user_context: str = "",
                                domain: str = "general",
                                additional_guidelines: str = "",
                                max_relations: int = 20) -> Dict[str, Any]:
        """
        관계 추론을 위한 최적화된 프롬프트 생성
        
        Args:
            text: 원본 텍스트
            entities: 추출된 엔티티 리스트
            user_context: 사용자 제공 컨텍스트
            domain: 도메인 유형
            additional_guidelines: 추가 가이드라인
            max_relations: 최대 관계 수
            
        Returns:
            프롬프트 생성 결과
        """
        try:
            # 도메인 템플릿 선택
            domain_template = self.domain_templates.get(domain, self.domain_templates["general"])
            
            # 엔티티 타입 분석
            entity_types = {}
            entity_pairs = []
            
            for entity in entities:
                entity_type = entity.get('type', 'UNKNOWN')
                if entity_type not in entity_types:
                    entity_types[entity_type] = []
                entity_types[entity_type].append(entity)
            
            # 엔티티 쌍 생성 (모든 조합)
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i != j:  # 자기 자신과의 관계는 제외
                        pair_key = f"{entity1.get('type', 'UNKNOWN')}-{entity2.get('type', 'UNKNOWN')}"
                        entity_pairs.append({
                            'entity1': entity1,
                            'entity2': entity2,
                            'pair_type': pair_key
                        })
            
            # 관계 패턴 추천
            recommended_relations = self._get_recommended_relations(entity_types, domain_template)
            
            # 기본 프롬프트 생성
            base_prompt = self._create_base_prompt(
                text, entities, domain_template, recommended_relations,
                user_context, additional_guidelines, max_relations
            )
            
            # LLM을 사용한 프롬프트 최적화 (선택적)
            optimized_prompt = base_prompt
            if self.llm_engine:
                optimized_prompt = self._optimize_prompt_with_llm(
                    base_prompt, text, entities, user_context
                )
            
            return {
                'success': True,
                'base_prompt': base_prompt,
                'optimized_prompt': optimized_prompt,
                'domain': domain,
                'domain_info': domain_template,
                'entity_types': entity_types,
                'entity_pairs_count': len(entity_pairs),
                'recommended_relations': recommended_relations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating relation prompt: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_recommended_relations(self, entity_types: Dict, domain_template: Dict) -> List[str]:
        """엔티티 타입 기반 추천 관계 생성"""
        recommended = set(domain_template['common_relations'])
        
        # 엔티티 타입 기반 패턴 매칭
        for type1 in entity_types.keys():
            for type2 in entity_types.keys():
                if type1 != type2:
                    pair_key = f"{type1}-{type2}"
                    if pair_key in domain_template['relation_patterns']:
                        recommended.update(domain_template['relation_patterns'][pair_key])
        
        return sorted(list(recommended))
    
    def _create_base_prompt(self, text: str, entities: List[Dict], domain_template: Dict,
                           recommended_relations: List[str], user_context: str,
                           additional_guidelines: str, max_relations: int) -> str:
        """기본 관계 추론 프롬프트 생성"""
        
        # 엔티티 정보 포맷팅
        entity_info = []
        for entity in entities:
            entity_info.append(f"- ID: {entity['id']}, 텍스트: \"{entity['text']}\", 타입: {entity['type']}")
        
        # 관계 타입 설명
        relation_descriptions = {
            "STUDIES": "A가 B를 연구/조사함",
            "ANALYZES": "A가 B를 분석함",
            "COMPARES": "A가 B와 비교됨",
            "USES": "A가 B를 사용/활용함",
            "MEASURES": "A가 B를 측정함",
            "EVALUATES": "A가 B를 평가함",
            "IMPLEMENTS": "A가 B를 구현함",
            "IS_TYPE_OF": "A가 B의 일종임",
            "PART_OF": "A가 B의 일부임",
            "CONTAINS": "A가 B를 포함함",
            "DESCRIBES": "A가 B를 설명/기술함",
            "REFERS_TO": "A가 B를 참조/언급함",
            "RELATED_TO": "A가 B와 관련됨",
            "SIMILAR_TO": "A가 B와 유사함",
            "BETTER_THAN": "A가 B보다 우수함",
            "DEPENDS_ON": "A가 B에 의존함",
            "CONNECTS_TO": "A가 B와 연결됨",
            "MANAGES": "A가 B를 관리함",
            "REPORTS_TO": "A가 B에게 보고함"
        }
        
        relation_list = []
        for rel in recommended_relations:
            desc = relation_descriptions.get(rel, f"A가 B와 {rel.lower().replace('_', ' ')} 관계")
            relation_list.append(f"- {rel}: {desc}")
        
        prompt = f"""# 엔티티 관계 추론 작업

## 📄 문서 정보
**도메인**: {domain_template['name']} ({domain_template['description']})
**원본 텍스트**: 
```
{text}
```

## 🎯 추출된 엔티티들
{chr(10).join(entity_info)}

## 💡 사용자 제공 컨텍스트
{user_context if user_context.strip() else "없음"}

## 📋 작업 지시사항
위 텍스트에서 추출된 엔티티들 사이의 의미적 관계를 분석하여 최대 {max_relations}개의 관계를 찾아주세요.

### 🔗 추천 관계 타입들
{chr(10).join(relation_list)}

### 📏 관계 추론 원칙
1. **텍스트 기반**: 반드시 원본 텍스트에 명시적으로 드러나거나 강하게 암시되는 관계만 추론
2. **신뢰도 평가**: 각 관계에 대해 0.0~1.0 사이의 신뢰도 점수 부여
3. **중요도 우선**: 문서의 주요 내용과 관련된 관계를 우선 선택
4. **중복 방지**: 동일한 엔티티 쌍에 대해 가장 적절한 하나의 관계만 선택
5. **방향성 고려**: 관계의 방향성이 중요한 경우 subject와 object 순서 주의

{f"### 📌 추가 가이드라인{chr(10)}{additional_guidelines}" if additional_guidelines.strip() else ""}

## 📤 출력 형식
다음 JSON 형식으로만 응답해주세요:
```json
[
    {{
        "subject": "엔티티_ID1",
        "type": "관계_타입",
        "object": "엔티티_ID2",
        "confidence": 0.85,
        "explanation": "관계 설정 근거 (텍스트의 특정 부분 인용)",
        "text_evidence": "관계를 뒷받침하는 원문 구절"
    }}
]
```

**중요**: JSON 배열 형식만 출력하고, 다른 텍스트는 포함하지 마세요."""

        return prompt
    
    def _optimize_prompt_with_llm(self, base_prompt: str, text: str, 
                                entities: List[Dict], user_context: str) -> str:
        """LLM을 사용한 프롬프트 최적화"""
        try:
            optimization_prompt = f"""
다음 관계 추론 프롬프트를 더 효과적이고 정확한 결과를 얻을 수 있도록 개선해주세요.

**원본 텍스트 특성**: {len(text)} 글자, {len(entities)}개 엔티티
**사용자 컨텍스트**: {user_context or "없음"}

**개선할 프롬프트**:
{base_prompt}

**개선 요청사항**:
1. 더 구체적이고 명확한 지시사항 제공
2. 텍스트의 특성에 맞는 관계 타입 강조
3. 오류를 방지할 수 있는 추가 안내 포함
4. JSON 출력 형식의 명확성 향상

개선된 프롬프트만 출력해주세요:
"""
            
            messages = [{"role": "user", "content": optimization_prompt}]
            response_generator = self.llm_engine.chat_completion(messages, stream=False)
            
            optimized_text = ""
            for response in response_generator:
                if response.get('type') == 'content':
                    optimized_text += response.get('text', '')
            
            # 최적화된 프롬프트 검증
            if len(optimized_text.strip()) > len(base_prompt) * 0.5:  # 최소 길이 체크
                return optimized_text.strip()
            else:
                logger.warning("Optimized prompt too short, using base prompt")
                return base_prompt
                
        except Exception as e:
            logger.warning(f"Failed to optimize prompt with LLM: {e}")
            return base_prompt


class RelationEvaluatorAgent:
    """
    관계 추론 결과 평가를 위한 Sub-agent
    LLM이 추론한 관계들의 품질을 평가하고 개선점을 제시
    """
    
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine
        
        # 평가 기준
        self.evaluation_criteria = {
            "text_support": {
                "name": "텍스트 근거성",
                "description": "관계가 원본 텍스트에 의해 뒷받침되는 정도",
                "weight": 0.3
            },
            "logical_consistency": {
                "name": "논리적 일관성",
                "description": "관계가 논리적으로 일관되고 모순이 없는 정도",
                "weight": 0.25
            },
            "semantic_accuracy": {
                "name": "의미적 정확성",
                "description": "관계 타입이 엔티티 간의 실제 관계를 정확히 표현하는 정도",
                "weight": 0.25
            },
            "relevance": {
                "name": "관련성",
                "description": "관계가 문서의 주요 내용과 얼마나 관련이 있는지",
                "weight": 0.2
            }
        }
        
        # 품질 임계값
        self.quality_thresholds = {
            "excellent": 0.85,
            "good": 0.7,
            "acceptable": 0.55,
            "poor": 0.0
        }
    
    def evaluate_relations(self, 
                          relations: List[Dict], 
                          text: str, 
                          entities: List[Dict],
                          user_context: str = "",
                          domain: str = "general") -> Dict[str, Any]:
        """
        추론된 관계들의 품질 평가
        
        Args:
            relations: 추론된 관계 리스트
            text: 원본 텍스트
            entities: 엔티티 리스트
            user_context: 사용자 컨텍스트
            domain: 도메인 유형
            
        Returns:
            평가 결과
        """
        try:
            if not relations:
                return {
                    'success': True,
                    'overall_score': 0.0,
                    'quality_level': 'poor',
                    'relation_scores': [],
                    'recommendations': ["관계가 추론되지 않았습니다. 프롬프트를 개선하거나 텍스트를 확인해주세요."],
                    'approved_relations': [],
                    'rejected_relations': [],
                    'needs_improvement': True,
                    'timestamp': datetime.now().isoformat()
                }
            
            # 개별 관계 평가
            relation_evaluations = []
            approved_relations = []
            rejected_relations = []
            
            for i, relation in enumerate(relations):
                evaluation = self._evaluate_single_relation(
                    relation, text, entities, user_context
                )
                evaluation['relation_index'] = i
                relation_evaluations.append(evaluation)
                
                if evaluation['approved']:
                    approved_relations.append(relation)
                else:
                    rejected_relations.append({
                        'relation': relation,
                        'rejection_reason': evaluation['issues']
                    })
            
            # 전체 품질 점수 계산
            if relation_evaluations:
                overall_score = sum(eval['score'] for eval in relation_evaluations) / len(relation_evaluations)
            else:
                overall_score = 0.0
            
            # 품질 수준 결정
            quality_level = self._determine_quality_level(overall_score)
            
            # LLM 기반 추가 평가 (선택적)
            llm_evaluation = {}
            if self.llm_engine:
                llm_evaluation = self._evaluate_with_llm(
                    relations, text, entities, user_context, relation_evaluations
                )
            
            # 개선 추천사항 생성
            recommendations = self._generate_recommendations(
                relation_evaluations, overall_score, quality_level, llm_evaluation
            )
            
            # 개선 필요 여부 결정
            needs_improvement = overall_score < self.quality_thresholds['good']
            
            return {
                'success': True,
                'overall_score': overall_score,
                'quality_level': quality_level,
                'relation_scores': relation_evaluations,
                'approved_relations': approved_relations,
                'rejected_relations': rejected_relations,
                'recommendations': recommendations,
                'llm_evaluation': llm_evaluation,
                'needs_improvement': needs_improvement,
                'evaluation_criteria': self.evaluation_criteria,
                'statistics': {
                    'total_relations': len(relations),
                    'approved_count': len(approved_relations),
                    'rejected_count': len(rejected_relations),
                    'approval_rate': len(approved_relations) / len(relations) if relations else 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating relations: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _evaluate_single_relation(self, relation: Dict, text: str, 
                                 entities: List[Dict], user_context: str) -> Dict[str, Any]:
        """개별 관계 평가"""
        
        # 필수 필드 검증
        required_fields = ['subject', 'type', 'object']
        missing_fields = [field for field in required_fields if field not in relation]
        
        if missing_fields:
            return {
                'score': 0.0,
                'approved': False,
                'issues': [f"필수 필드 누락: {', '.join(missing_fields)}"],
                'criteria_scores': {criterion: 0.0 for criterion in self.evaluation_criteria}
            }
        
        # 엔티티 존재 검증
        entity_ids = [e['id'] for e in entities]
        if relation['subject'] not in entity_ids:
            return {
                'score': 0.0,
                'approved': False,
                'issues': [f"존재하지 않는 subject 엔티티: {relation['subject']}"],
                'criteria_scores': {criterion: 0.0 for criterion in self.evaluation_criteria}
            }
        
        if relation['object'] not in entity_ids:
            return {
                'score': 0.0,
                'approved': False,
                'issues': [f"존재하지 않는 object 엔티티: {relation['object']}"],
                'criteria_scores': {criterion: 0.0 for criterion in self.evaluation_criteria}
            }
        
        # 각 기준별 평가
        criteria_scores = {}
        issues = []
        
        # 1. 텍스트 근거성 평가
        text_support_score = self._evaluate_text_support(relation, text)
        criteria_scores['text_support'] = text_support_score
        
        if text_support_score < 0.3:
            issues.append("텍스트에서 관계를 뒷받침하는 근거가 부족함")
        
        # 2. 논리적 일관성 평가
        logical_score = self._evaluate_logical_consistency(relation, entities)
        criteria_scores['logical_consistency'] = logical_score
        
        if logical_score < 0.3:
            issues.append("관계가 논리적으로 일관성이 부족함")
        
        # 3. 의미적 정확성 평가
        semantic_score = self._evaluate_semantic_accuracy(relation, entities)
        criteria_scores['semantic_accuracy'] = semantic_score
        
        if semantic_score < 0.3:
            issues.append("관계 타입이 엔티티 간의 실제 관계를 부정확하게 표현함")
        
        # 4. 관련성 평가
        relevance_score = self._evaluate_relevance(relation, text, user_context)
        criteria_scores['relevance'] = relevance_score
        
        if relevance_score < 0.3:
            issues.append("문서의 주요 내용과 관련성이 낮음")
        
        # 가중 평균 점수 계산
        total_score = sum(
            score * self.evaluation_criteria[criterion]['weight']
            for criterion, score in criteria_scores.items()
        )
        
        # 승인 여부 결정
        approved = total_score >= self.quality_thresholds['acceptable'] and len(issues) == 0
        
        return {
            'score': total_score,
            'approved': approved,
            'issues': issues,
            'criteria_scores': criteria_scores,
            'confidence': relation.get('confidence', 0.5)
        }
    
    def _evaluate_text_support(self, relation: Dict, text: str) -> float:
        """텍스트 근거성 평가"""
        try:
            # 관계에 text_evidence가 있는지 확인
            text_evidence = relation.get('text_evidence', '')
            explanation = relation.get('explanation', '')
            
            # 텍스트 증거가 실제로 원문에 있는지 확인
            if text_evidence and text_evidence.strip() in text:
                return 0.8
            
            # explanation이 구체적인지 확인
            if explanation and len(explanation.strip()) > 10:
                return 0.6
            
            # confidence 값이 있으면 참고
            confidence = relation.get('confidence', 0.5)
            if confidence > 0.7:
                return 0.5
            
            return 0.3
            
        except Exception:
            return 0.2
    
    def _evaluate_logical_consistency(self, relation: Dict, entities: List[Dict]) -> float:
        """논리적 일관성 평가"""
        try:
            # 자기 자신과의 관계 체크
            if relation['subject'] == relation['object']:
                return 0.1
            
            # 엔티티 타입 기반 관계의 적절성 체크
            subject_entity = next(e for e in entities if e['id'] == relation['subject'])
            object_entity = next(e for e in entities if e['id'] == relation['object'])
            
            subject_type = subject_entity.get('type', 'UNKNOWN')
            object_type = object_entity.get('type', 'UNKNOWN')
            relation_type = relation['type']
            
            # 일부 관계 타입에 대한 기본 논리 체크
            if relation_type == 'IS_TYPE_OF' and subject_type == object_type:
                return 0.3  # 같은 타입끼리는 IS_TYPE_OF가 부적절할 수 있음
            
            if relation_type in ['PART_OF', 'CONTAINS'] and subject_type == object_type:
                return 0.6  # 같은 타입끼리도 가능하지만 주의 필요
            
            return 0.8  # 기본적으로는 논리적으로 일관된 것으로 가정
            
        except Exception:
            return 0.4
    
    def _evaluate_semantic_accuracy(self, relation: Dict, entities: List[Dict]) -> float:
        """의미적 정확성 평가"""
        try:
            relation_type = relation['type']
            confidence = relation.get('confidence', 0.5)
            
            # 관계 타입이 표준적인지 확인
            standard_relations = [
                'STUDIES', 'ANALYZES', 'COMPARES', 'USES', 'MEASURES', 'EVALUATES',
                'IMPLEMENTS', 'IS_TYPE_OF', 'PART_OF', 'CONTAINS', 'DESCRIBES',
                'REFERS_TO', 'RELATED_TO', 'SIMILAR_TO', 'DEPENDS_ON'
            ]
            
            if relation_type in standard_relations:
                return min(0.9, 0.5 + confidence * 0.4)
            else:
                # 비표준 관계 타입의 경우 낮은 점수
                return min(0.6, confidence * 0.6)
                
        except Exception:
            return 0.4
    
    def _evaluate_relevance(self, relation: Dict, text: str, user_context: str) -> float:
        """관련성 평가"""
        try:
            # 사용자 컨텍스트와의 관련성
            if user_context:
                subject_entity = relation['subject']
                object_entity = relation['object']
                
                # 컨텍스트에 엔티티들이 언급되는지 확인
                context_mentions = 0
                if subject_entity in user_context:
                    context_mentions += 1
                if object_entity in user_context:
                    context_mentions += 1
                
                if context_mentions == 2:
                    return 0.9
                elif context_mentions == 1:
                    return 0.7
            
            # confidence 기반 관련성
            confidence = relation.get('confidence', 0.5)
            return min(0.8, 0.3 + confidence * 0.5)
            
        except Exception:
            return 0.5
    
    def _determine_quality_level(self, score: float) -> str:
        """점수 기반 품질 수준 결정"""
        for level, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return level
        return 'poor'
    
    def _evaluate_with_llm(self, relations: List[Dict], text: str, 
                          entities: List[Dict], user_context: str,
                          relation_evaluations: List[Dict]) -> Dict[str, Any]:
        """LLM을 사용한 추가 평가"""
        try:
            evaluation_prompt = f"""
다음 관계 추론 결과를 평가해주세요.

**원본 텍스트**:
{text}

**추론된 관계들**:
{json.dumps(relations, ensure_ascii=False, indent=2)}

**사용자 컨텍스트**: {user_context or "없음"}

각 관계에 대해 다음 항목을 평가해주세요:
1. 텍스트 근거의 충분성 (1-10점)
2. 관계 타입의 적절성 (1-10점)
3. 전체적인 유용성 (1-10점)

또한 다음을 제공해주세요:
- 가장 좋은 관계 3개
- 가장 문제가 있는 관계 3개
- 전체적인 품질 평가
- 개선 제안사항

JSON 형식으로 응답해주세요:
```json
{{
    "overall_assessment": "전체 평가 코멘트",
    "best_relations": ["관계1 설명", "관계2 설명", "관계3 설명"],
    "problematic_relations": ["문제 관계1", "문제 관계2", "문제 관계3"],
    "improvement_suggestions": ["제안1", "제안2", "제안3"],
    "quality_score": 7.5
}}
```
"""
            
            messages = [{"role": "user", "content": evaluation_prompt}]
            response_generator = self.llm_engine.chat_completion(messages, stream=False)
            
            response_text = ""
            for response in response_generator:
                if response.get('type') == 'content':
                    response_text += response.get('text', '')
            
            # JSON 파싱
            try:
                # JSON 부분만 추출
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    # JSON 블록이 없으면 전체에서 JSON 찾기
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
                    else:
                        raise ValueError("No JSON found in LLM response")
                
                llm_eval = json.loads(json_text)
                return llm_eval
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM evaluation response: {e}")
                return {
                    'error': 'Failed to parse LLM response',
                    'raw_response': response_text[:500]
                }
                
        except Exception as e:
            logger.warning(f"Failed to evaluate with LLM: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, relation_evaluations: List[Dict], 
                                overall_score: float, quality_level: str,
                                llm_evaluation: Dict) -> List[str]:
        """개선 추천사항 생성"""
        recommendations = []
        
        # 전체 품질 기반 추천
        if quality_level == 'poor':
            recommendations.append("전체적인 관계 추론 품질이 낮습니다. 더 구체적인 컨텍스트를 제공하거나 프롬프트를 개선해보세요.")
        elif quality_level == 'acceptable':
            recommendations.append("관계 추론 품질이 보통 수준입니다. 일부 관계의 근거를 더 명확히 하면 개선될 것 같습니다.")
        elif quality_level == 'good':
            recommendations.append("관계 추론 품질이 좋습니다. 소수의 관계만 개선하면 더욱 완성도가 높아질 것입니다.")
        else:
            recommendations.append("관계 추론 품질이 우수합니다!")
        
        # 개별 관계 기반 추천
        low_score_count = sum(1 for eval in relation_evaluations if eval['score'] < 0.5)
        if low_score_count > 0:
            recommendations.append(f"{low_score_count}개의 관계가 낮은 점수를 받았습니다. 해당 관계들을 검토해보세요.")
        
        # 일반적인 문제 패턴 확인
        common_issues = {}
        for eval in relation_evaluations:
            for issue in eval['issues']:
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        if common_issues:
            most_common = max(common_issues.items(), key=lambda x: x[1])
            if most_common[1] > 1:
                recommendations.append(f"공통 문제: {most_common[0]} (총 {most_common[1]}건)")
        
        # LLM 평가 기반 추천
        if llm_evaluation and 'improvement_suggestions' in llm_evaluation:
            recommendations.extend(llm_evaluation['improvement_suggestions'])
        
        return recommendations[:10]  # 최대 10개로 제한


class IterativeRelationInferenceAgent:
    """
    반복적 관계 추론 개선을 위한 통합 Agent
    RelationPromptAgent와 RelationEvaluatorAgent를 조합하여 고품질 관계 추론
    """
    
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine
        self.prompt_agent = RelationPromptAgent(llm_engine)
        self.evaluator_agent = RelationEvaluatorAgent(llm_engine)
        
        self.max_iterations = 3  # 최대 반복 횟수
        self.target_score = 0.75  # 목표 품질 점수
    
    def infer_relations_iteratively(self, 
                                   text: str,
                                   entities: List[Dict],
                                   user_context: str = "",
                                   domain: str = "general",
                                   additional_guidelines: str = "",
                                   max_relations: int = 20) -> Dict[str, Any]:
        """
        반복적 관계 추론 실행
        
        Returns:
            최종 관계 추론 결과
        """
        try:
            iteration_results = []
            best_result = None
            best_score = 0.0
            
            for iteration in range(self.max_iterations):
                logger.info(f"Starting relation inference iteration {iteration + 1}/{self.max_iterations}")
                
                # 1. 프롬프트 생성 (이전 결과 기반 개선)
                improvement_context = ""
                if iteration > 0 and iteration_results:
                    prev_evaluation = iteration_results[-1]['evaluation']
                    improvement_context = f"이전 시도의 문제점을 개선해주세요: {', '.join(prev_evaluation['recommendations'][:3])}"
                
                prompt_result = self.prompt_agent.generate_relation_prompt(
                    text=text,
                    entities=entities,
                    user_context=user_context,
                    domain=domain,
                    additional_guidelines=additional_guidelines + "\n" + improvement_context,
                    max_relations=max_relations
                )
                
                if not prompt_result['success']:
                    continue
                
                # 2. LLM으로 관계 추론
                relations = self._infer_relations_with_llm(
                    prompt_result['optimized_prompt'],
                    text,
                    entities
                )
                
                if not relations:
                    continue
                
                # 3. 결과 평가
                evaluation_result = self.evaluator_agent.evaluate_relations(
                    relations=relations,
                    text=text,
                    entities=entities,
                    user_context=user_context,
                    domain=domain
                )
                
                if not evaluation_result['success']:
                    continue
                
                # 4. 결과 저장
                iteration_result = {
                    'iteration': iteration + 1,
                    'prompt_result': prompt_result,
                    'relations': relations,
                    'evaluation': evaluation_result,
                    'score': evaluation_result['overall_score']
                }
                iteration_results.append(iteration_result)
                
                # 5. 최고 결과 업데이트
                current_score = evaluation_result['overall_score']
                if current_score > best_score:
                    best_result = iteration_result
                    best_score = current_score
                
                # 6. 목표 점수 달성 시 조기 종료
                if current_score >= self.target_score:
                    logger.info(f"Target quality score {self.target_score} achieved at iteration {iteration + 1}")
                    break
                
                # 7. 개선이 없으면 조기 종료
                if iteration > 0 and current_score <= iteration_results[-2]['score'] - 0.05:
                    logger.info(f"No significant improvement in iteration {iteration + 1}, stopping")
                    break
            
            # 최종 결과 생성
            if best_result:
                final_relations = best_result['evaluation']['approved_relations']
                
                return {
                    'success': True,
                    'final_relations': final_relations,
                    'best_iteration': best_result['iteration'],
                    'best_score': best_score,
                    'total_iterations': len(iteration_results),
                    'iteration_history': [
                        {
                            'iteration': r['iteration'],
                            'score': r['score'],
                            'relations_count': len(r['relations']),
                            'approved_count': len(r['evaluation']['approved_relations'])
                        }
                        for r in iteration_results
                    ],
                    'final_evaluation': best_result['evaluation'],
                    'improvement_achieved': best_score > (iteration_results[0]['score'] if iteration_results else 0),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'No successful iterations completed',
                    'iteration_attempts': len(iteration_results),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in iterative relation inference: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _infer_relations_with_llm(self, prompt: str, text: str, entities: List[Dict]) -> List[Dict]:
        """LLM을 사용한 관계 추론"""
        try:
            if not self.llm_engine:
                logger.warning("No LLM engine available for relation inference")
                return []
            
            messages = [{"role": "user", "content": prompt}]
            response_generator = self.llm_engine.chat_completion(messages, stream=False)
            
            response_text = ""
            for response in response_generator:
                if response.get('type') == 'content':
                    response_text += response.get('text', '')
            
            # JSON 파싱 시도
            relations = self._parse_relations_response(response_text)
            
            # 관계 유효성 검증
            valid_relations = []
            entity_ids = [e['id'] for e in entities]
            
            for relation in relations:
                if (isinstance(relation, dict) and 
                    'subject' in relation and 'type' in relation and 'object' in relation and
                    relation['subject'] in entity_ids and relation['object'] in entity_ids):
                    valid_relations.append(relation)
            
            logger.info(f"Successfully parsed {len(valid_relations)} valid relations from LLM response")
            return valid_relations
            
        except Exception as e:
            logger.error(f"Error in LLM relation inference: {e}")
            return []
    
    def _parse_relations_response(self, response_text: str) -> List[Dict]:
        """LLM 응답에서 관계 파싱"""
        try:
            # 1. JSON 블록 찾기
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # 2. JSON 배열 찾기
                json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    raise ValueError("No JSON array found in response")
            
            # 3. JSON 파싱
            relations = json.loads(json_text)
            
            if not isinstance(relations, list):
                raise ValueError("Response is not a JSON array")
            
            return relations
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse relations JSON: {e}")
            
            # 4. 대안: 정규식으로 관계 패턴 찾기
            try:
                return self._extract_relations_with_regex(response_text)
            except Exception as e2:
                logger.error(f"Failed to extract relations with regex: {e2}")
                return []
    
    def _extract_relations_with_regex(self, text: str) -> List[Dict]:
        """정규식을 사용한 관계 추출 (fallback)"""
        relations = []
        
        # subject, type, object 패턴 찾기
        pattern = r'"subject"\s*:\s*"([^"]+)".*?"type"\s*:\s*"([^"]+)".*?"object"\s*:\s*"([^"]+)"'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            subject, rel_type, obj = match
            relations.append({
                'subject': subject.strip(),
                'type': rel_type.strip(),
                'object': obj.strip(),
                'confidence': 0.5,  # 기본값
                'explanation': 'Extracted via regex fallback'
            })
        
        return relations