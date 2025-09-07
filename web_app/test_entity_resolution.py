#!/usr/bin/env python3
"""
Entity Resolution System 통합 테스트 스크립트
"""

import json
import logging
import sys
import os
from typing import Dict, List, Any

# 현재 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 앱 컨텍스트 설정
from app import create_app
from app.entity_resolution_agents import EntityResolutionAgent, EntityResolutionValidator
from app.knowledge_graph_agents import KnowledgeGraphGenerator

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_entities() -> List[Dict]:
    """테스트용 엔티티 데이터 생성"""
    return [
        {
            'id': 'entity_1',
            'text': 'GPT-4',
            'type': 'MODEL',
            'start': 10,
            'end': 15,
            'attributes': {'entity_type': 'MODEL', 'confidence': 0.95}
        },
        {
            'id': 'entity_2', 
            'text': 'GPT4',  # 동일한 모델, 다른 표기
            'type': 'MODEL',
            'start': 50,
            'end': 54,
            'attributes': {'entity_type': 'MODEL', 'confidence': 0.90}
        },
        {
            'id': 'entity_3',
            'text': 'gpt-4',  # 동일한 모델, 소문자
            'type': 'MODEL', 
            'start': 80,
            'end': 85,
            'attributes': {'entity_type': 'MODEL', 'confidence': 0.85}
        },
        {
            'id': 'entity_4',
            'text': 'Llama2',
            'type': 'MODEL',
            'start': 120,
            'end': 126,
            'attributes': {'entity_type': 'MODEL', 'confidence': 0.92}
        },
        {
            'id': 'entity_5',
            'text': 'LLaMA-2',  # 동일한 모델, 다른 표기
            'type': 'MODEL',
            'start': 150,
            'end': 157,
            'attributes': {'entity_type': 'MODEL', 'confidence': 0.88}
        },
        {
            'id': 'entity_6',
            'text': 'Temperature',  # 매개변수
            'type': 'TECHNOLOGY_PARAMETER',
            'start': 200,
            'end': 211,
            'attributes': {'entity_type': 'TECHNOLOGY_PARAMETER', 'confidence': 0.90}
        },
        {
            'id': 'entity_7',
            'text': 'temperature',  # 소문자 동일 매개변수
            'type': 'TECHNOLOGY_PARAMETER',
            'start': 250,
            'end': 261,
            'attributes': {'entity_type': 'TECHNOLOGY_PARAMETER', 'confidence': 0.88}
        },
        {
            'id': 'entity_8',
            'text': 'Machine Learning',  # 다른 타입
            'type': 'CONCEPT',
            'start': 300,
            'end': 316,
            'attributes': {'entity_type': 'CONCEPT', 'confidence': 0.93}
        }
    ]


def test_entity_resolution_agent():
    """EntityResolutionAgent 단독 테스트"""
    logger.info("=== Testing EntityResolutionAgent ===")
    
    # 테스트 데이터 생성
    test_entities = create_test_entities()
    logger.info(f"Created {len(test_entities)} test entities")
    
    # EntityResolutionAgent 초기화
    resolution_agent = EntityResolutionAgent()
    
    # 엔티티 해결 실행
    resolution_result = resolution_agent.resolve_entities(
        test_entities,
        resolution_config={
            'similarity_threshold': 0.8,
            'use_fuzzy_matching': True,
            'use_semantic_similarity': False,  # LLM 엔진 없음
            'merge_strategy': 'keep_first'
        }
    )
    
    # 결과 분석
    if resolution_result.get('success'):
        resolved_entities = resolution_result.get('resolved_entities', [])
        entity_mapping = resolution_result.get('entity_mapping', {})
        merged_groups = resolution_result.get('merged_groups', [])
        stats = resolution_result.get('statistics', {})
        
        logger.info(f"✅ Resolution successful!")
        logger.info(f"   Original entities: {stats.get('total_original', 0)}")
        logger.info(f"   Resolved entities: {stats.get('total_resolved', 0)}")
        logger.info(f"   Reduction: {stats.get('reduction_percentage', 0):.1f}%")
        logger.info(f"   Merged groups: {len(merged_groups)}")
        
        # 병합된 그룹 출력
        for i, group in enumerate(merged_groups):
            group_texts = [entity.get('text', entity.get('entity_text', '')) 
                          for entity in test_entities if entity['id'] in group]
            logger.info(f"   Group {i+1}: {group_texts}")
        
        return resolution_result
    else:
        logger.error(f"❌ Resolution failed: {resolution_result.get('error')}")
        return None


def test_entity_resolution_validator(resolution_result: Dict):
    """EntityResolutionValidator 테스트"""
    logger.info("=== Testing EntityResolutionValidator ===")
    
    if not resolution_result:
        logger.error("❌ No resolution result to validate")
        return None
    
    # EntityResolutionValidator 초기화
    validator = EntityResolutionValidator()
    
    # 검증 실행
    validation_result = validator.validate_resolution(resolution_result)
    
    if validation_result.get('success'):
        logger.info("✅ Validation successful!")
        validation_score = validation_result.get('overall_score', 0)
        logger.info(f"   Overall score: {validation_score:.2f}")
        
        # 개별 규칙 결과 출력
        rule_results = validation_result.get('rule_results', [])
        for rule in rule_results:
            status = "✅" if rule['passed'] else "❌"
            logger.info(f"   {status} {rule['rule']}: {rule['message']}")
            
            if rule.get('issues'):
                for issue in rule['issues'][:3]:  # 첫 3개만 출력
                    logger.info(f"      - {issue}")
        
        return validation_result
    else:
        logger.error(f"❌ Validation failed: {validation_result.get('error')}")
        return None


def test_integrated_knowledge_graph():
    """Knowledge Graph 통합 테스트"""
    logger.info("=== Testing Integrated Knowledge Graph Generation ===")
    
    # LLM-IE 형식의 테스트 데이터 생성
    extraction_data = {
        "doc_id": "entity_resolution_test",
        "text": "이 연구에서는 GPT-4, GPT4, gpt-4 모델과 Llama2, LLaMA-2 모델을 비교했습니다. Temperature, temperature 매개변수가 성능에 미치는 영향을 Machine Learning 관점에서 분석했습니다.",
        "frames": [
            {
                "frame_id": "0",
                "start": 10,
                "end": 15,
                "entity_text": "GPT-4",
                "attr": {"entity_type": "MODEL"}
            },
            {
                "frame_id": "1", 
                "start": 17,
                "end": 21,
                "entity_text": "GPT4",
                "attr": {"entity_type": "MODEL"}
            },
            {
                "frame_id": "2",
                "start": 23,
                "end": 28,
                "entity_text": "gpt-4", 
                "attr": {"entity_type": "MODEL"}
            },
            {
                "frame_id": "3",
                "start": 35,
                "end": 41,
                "entity_text": "Llama2",
                "attr": {"entity_type": "MODEL"}
            },
            {
                "frame_id": "4",
                "start": 43,
                "end": 50,
                "entity_text": "LLaMA-2",
                "attr": {"entity_type": "MODEL"}
            },
            {
                "frame_id": "5",
                "start": 65,
                "end": 76,
                "entity_text": "Temperature",
                "attr": {"entity_type": "TECHNOLOGY_PARAMETER"}
            },
            {
                "frame_id": "6",
                "start": 78,
                "end": 89,
                "entity_text": "temperature",
                "attr": {"entity_type": "TECHNOLOGY_PARAMETER"}
            },
            {
                "frame_id": "7",
                "start": 110,
                "end": 126,
                "entity_text": "Machine Learning",
                "attr": {"entity_type": "CONCEPT"}
            }
        ],
        "relations": []
    }
    
    # Knowledge Graph Generator 초기화 (LLM 엔진 없이)
    kg_generator = KnowledgeGraphGenerator(llm_engine=None)
    
    # 지식그래프 생성 (엔티티 해결 활성화)
    logger.info("Testing with Entity Resolution ENABLED...")
    result_with_resolution = kg_generator.generate_knowledge_graph(
        extraction_data=extraction_data,
        enable_entity_resolution=True
    )
    
    # 지식그래프 생성 (엔티티 해결 비활성화)  
    logger.info("Testing with Entity Resolution DISABLED...")
    result_without_resolution = kg_generator.generate_knowledge_graph(
        extraction_data=extraction_data,
        enable_entity_resolution=False
    )
    
    # 결과 비교
    if result_with_resolution.get('success') and result_without_resolution.get('success'):
        entities_with = result_with_resolution.get('total_entities', 0)
        entities_without = result_without_resolution.get('total_entities', 0)
        
        logger.info("✅ Knowledge Graph Generation Results:")
        logger.info(f"   Without Entity Resolution: {entities_without} entities")
        logger.info(f"   With Entity Resolution: {entities_with} entities")
        logger.info(f"   Entity Reduction: {entities_without - entities_with} entities")
        
        if entities_with < entities_without:
            logger.info("✅ Entity resolution successfully reduced duplicate entities!")
        else:
            logger.warning("⚠️ Entity resolution did not reduce entities")
            
        return result_with_resolution, result_without_resolution
    else:
        logger.error("❌ Knowledge graph generation failed")
        return None, None


def test_error_handling():
    """오류 처리 테스트"""
    logger.info("=== Testing Error Handling ===")
    
    resolution_agent = EntityResolutionAgent()
    
    # 빈 엔티티 리스트 테스트
    result = resolution_agent.resolve_entities([])
    logger.info(f"Empty entities test: {'✅' if result.get('success') else '❌'}")
    
    # 잘못된 형식 엔티티 테스트
    invalid_entities = [{'invalid': 'data'}]
    result = resolution_agent.resolve_entities(invalid_entities)
    logger.info(f"Invalid entities test: {'✅' if not result.get('success') else '❌'}")
    
    # 극단적인 임계값 테스트
    test_entities = create_test_entities()[:2]
    result = resolution_agent.resolve_entities(
        test_entities,
        resolution_config={'similarity_threshold': 1.0}  # 완전 일치만
    )
    logger.info(f"High threshold test: {'✅' if result.get('success') else '❌'}")


def main():
    """메인 테스트 실행"""
    logger.info("🚀 Starting Entity Resolution System Tests")
    
    # Flask 앱 컨텍스트 생성
    app = create_app()
    with app.app_context():
        try:
            # 1. Entity Resolution Agent 테스트
            resolution_result = test_entity_resolution_agent()
            
            # 2. Entity Resolution Validator 테스트
            if resolution_result:
                test_entity_resolution_validator(resolution_result)
            
            # 3. 통합 Knowledge Graph 테스트
            test_integrated_knowledge_graph()
            
            # 4. 오류 처리 테스트
            test_error_handling()
            
            logger.info("🎉 All tests completed!")
            
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()