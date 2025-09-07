#!/usr/bin/env python3
"""
관계 생성 디버깅 스크립트
지식그래프에서 관계가 생성되지 않는 문제를 진단
"""

import sys
import os
import json
import logging

# 웹앱 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockLLMEngine:
    def __init__(self):
        self.call_count = 0
        
    def chat_completion(self, messages, stream=False):
        self.call_count += 1
        
        # 메시지 내용에 따라 다른 응답 생성
        last_message = messages[-1]['content'] if messages else ''
        
        if 'prompt improvement' in last_message.lower() or '프롬프트 개선' in last_message:
            # RelationPromptAgent용 응답
            response = {
                "success": True,
                "improved_prompt": "다음 텍스트에서 엔티티들 간의 의미적 관계를 분석하세요...",
                "improvements_made": ["도메인 특화 지침 추가", "관계 유형 명확화"],
                "domain_guidelines": ["학술 논문에서는 STUDIES, ANALYZES 관계가 중요"]
            }
            yield {'type': 'content', 'text': json.dumps(response, ensure_ascii=False)}
            
        elif 'evaluate' in last_message.lower() or '평가' in last_message:
            # RelationEvaluatorAgent용 응답
            response = {
                "relevance_score": 8,
                "accuracy_score": 7,
                "consistency_score": 9,
                "specificity_score": 6,
                "overall_score": 7.5,
                "feedback": "관계가 대체로 적절합니다.",
                "suggestions": ["관계의 맥락을 더 명확히 설명하세요"]
            }
            yield {'type': 'content', 'text': json.dumps(response, ensure_ascii=False)}
            
        else:
            # 관계 추론용 응답 (IterativeRelationInferenceAgent 파서가 기대하는 형식)
            relations_response = [
                {
                    "subject": "entity_1",
                    "relation_type": "USES",
                    "object": "entity_2", 
                    "confidence": 0.85,
                    "explanation": "딥러닝이 자연어처리에 사용됩니다."
                },
                {
                    "subject": "entity_2",
                    "relation_type": "IMPLEMENTS",
                    "object": "entity_3",
                    "confidence": 0.9,
                    "explanation": "자연어처리가 BERT 모델을 구현합니다."
                },
                {
                    "subject": "entity_1",
                    "relation_type": "RELATED_TO",
                    "object": "entity_3",
                    "confidence": 0.75,
                    "explanation": "딥러닝과 BERT 모델은 관련이 있습니다."
                }
            ]
            
            # 파서가 기대하는 형식으로 JSON 블록 포함
            response_text = f"다음은 분석된 관계들입니다:\n\n```json\n{json.dumps(relations_response, ensure_ascii=False, indent=2)}\n```"
            yield {'type': 'content', 'text': response_text}


def test_relation_creation():
    """관계 생성 프로세스 디버그 테스트"""
    
    print("=== 관계 생성 디버그 테스트 ===\n")
    
    # 테스트 데이터
    sample_frames = [
        {
            'frame_id': 'entity_1',
            'entity_text': '딥러닝',
            'start': 0,
            'end': 3,
            'attr': {'entity_type': 'TECHNOLOGY'}
        },
        {
            'frame_id': 'entity_2',
            'entity_text': '자연어처리',
            'start': 10,
            'end': 15,
            'attr': {'entity_type': 'FIELD'}
        },
        {
            'frame_id': 'entity_3',
            'entity_text': 'BERT 모델',
            'start': 20,
            'end': 27,
            'attr': {'entity_type': 'MODEL'}
        }
    ]
    
    sample_text = "딥러닝을 활용한 자연어처리 연구에서 BERT 모델을 사용하여 텍스트 분류 성능을 향상시켰습니다."
    
    extraction_data = {
        'doc_id': 'debug_test',
        'text': sample_text,
        'frames': sample_frames,
        'relations': []
    }
    
    try:
        from app.knowledge_graph_agents import KnowledgeGraphGenerator
        
        # Mock LLM 엔진으로 테스트
        mock_engine = MockLLMEngine()
        generator = KnowledgeGraphGenerator(llm_engine=mock_engine)
        
        print("1. 지식그래프 생성 시작 (Multi-agent 활성화)...")
        result = generator.generate_knowledge_graph(
            extraction_data=extraction_data,
            user_context='디버그 테스트',
            domain='research_paper',
            use_iterative_inference=True  # Multi-agent 시스템 사용
        )
        
        print(f"2. 생성 결과: {result.get('success')}")
        print(f"3. 총 트리플 수: {result.get('total_triples', 0)}")
        
        if 'statistics' in result:
            stats = result['statistics']
            print(f"4. 통계:")
            print(f"   - 엔티티 수: {stats.get('total_entities', 0)}")
            print(f"   - 관계 수: {stats.get('total_relations', 0)}")
            print(f"   - 문서 수: {stats.get('total_documents', 0)}")
        
        if 'visualization_data' in result:
            viz_data = result['visualization_data']
            print(f"5. 시각화 데이터:")
            print(f"   - 노드 수: {len(viz_data.get('nodes', []))}")
            print(f"   - 엣지 수: {len(viz_data.get('edges', []))}")
            
            # 노드와 엣지 상세 정보
            print("\n   노드들:")
            for node in viz_data.get('nodes', []):
                print(f"     - {node.get('label')} ({node.get('type')})")
            
            print("\n   엣지들:")
            edges = viz_data.get('edges', [])
            if edges:
                for edge in edges:
                    print(f"     - {edge.get('from')} --{edge.get('label')}--> {edge.get('to')} (신뢰도: {edge.get('confidence')})")
            else:
                print("     - 엣지가 없습니다!")
                
        # RDF 데이터 검사
        if 'rdf_formats' in result and 'turtle' in result['rdf_formats']:
            turtle_data = result['rdf_formats']['turtle']
            print(f"\n6. RDF Turtle 데이터 (처음 500자):")
            print(turtle_data[:500] + "..." if len(turtle_data) > 500 else turtle_data)
                
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}", exc_info=True)
        print(f"오류: {e}")


def test_individual_relation():
    """개별 관계 생성 테스트"""
    
    print("\n=== 개별 관계 생성 테스트 ===\n")
    
    try:
        from app.knowledge_graph_agents import KnowledgeGraphGenerator
        
        generator = KnowledgeGraphGenerator()
        
        # 엔티티 URI 딕셔너리 수동 생성
        entity_uris = {
            'entity_1': generator.ENTITY['entity_1_딥러닝'],
            'entity_2': generator.ENTITY['entity_2_자연어처리'],
            'entity_3': generator.ENTITY['entity_3_BERT_모델']
        }
        
        # 테스트 관계
        test_relation = {
            'subject': 'entity_1',
            'relation_type': 'USES',
            'object': 'entity_2',
            'confidence': 0.85,
            'explanation': '테스트 관계'
        }
        
        print("1. 관계 데이터:", test_relation)
        print("2. 엔티티 URIs:", {k: str(v) for k, v in entity_uris.items()})
        
        # 관계 생성 시도
        generator._create_relation_node(test_relation, entity_uris)
        
        # RDF 그래프에서 관계 확인
        from rdflib import RDF
        statements = list(generator.graph.subjects(RDF.type, RDF.Statement))
        print(f"3. 생성된 Statement 노드 수: {len(statements)}")
        
        for stmt in statements:
            subjects = list(generator.graph.objects(stmt, RDF.subject))
            predicates = list(generator.graph.objects(stmt, RDF.predicate))
            objects = list(generator.graph.objects(stmt, RDF.object))
            confidences = list(generator.graph.objects(stmt, generator.LLMIE.hasConfidence))
            
            print(f"   - Subject: {subjects[0] if subjects else 'None'}")
            print(f"   - Predicate: {predicates[0] if predicates else 'None'}")
            print(f"   - Object: {objects[0] if objects else 'None'}")
            print(f"   - Confidence: {confidences[0] if confidences else 'None'}")
            
    except Exception as e:
        logger.error(f"개별 관계 테스트 중 오류: {e}", exc_info=True)
        print(f"오류: {e}")


if __name__ == "__main__":
    test_relation_creation()
    test_individual_relation()