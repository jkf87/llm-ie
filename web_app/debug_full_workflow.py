#!/usr/bin/env python3
"""
전체 Multi-Agent 워크플로우 디버깅
IterativeRelationInferenceAgent의 전체 플로우를 단계별로 확인
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

class DetailedMockLLMEngine:
    def __init__(self):
        self.call_count = 0
        
    def chat_completion(self, messages, stream=False):
        self.call_count += 1
        
        # 마지막 메시지 확인
        last_message = messages[-1]['content'] if messages else ''
        
        print(f"\n=== LLM 호출 #{self.call_count} ===")
        print(f"메시지 길이: {len(last_message)} 문자")
        print(f"메시지 시작: {last_message[:150]}...")
        
        # 프롬프트 개선 요청인지 확인 (이것부터 확인해야 함)
        if ('프롬프트를 더 효과적이고' in last_message or 
            'prompt improvement' in last_message.lower() or
            '개선해주세요' in last_message):
            print("→ 프롬프트 개선 응답 반환")
            response = {
                "success": True,
                "improved_prompt": "엔티티들 간의 관계를 JSON 형태로 추론해주세요.",
                "improvements_made": ["명확성 향상"],
                "domain_guidelines": []
            }
            yield {'type': 'content', 'text': json.dumps(response, ensure_ascii=False)}
        
        # 관계 평가 요청인지 확인
        elif ('관계들의 품질을 평가' in last_message or 
              'evaluate the following relations' in last_message.lower()):
            print("→ 평가 응답 반환")
            response = {
                "relevance_score": 8,
                "accuracy_score": 9,
                "consistency_score": 8,
                "specificity_score": 7,
                "overall_score": 8.0,
                "feedback": "관계들이 잘 추론되었습니다.",
                "suggestions": []
            }
            yield {'type': 'content', 'text': json.dumps(response, ensure_ascii=False)}
        
        else:
            # 관계 추론 응답
            print("→ 관계 추론 응답 반환")
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
                }
            ]
            
            response_text = f"다음은 분석된 관계들입니다:\n\n```json\n{json.dumps(relations_response, ensure_ascii=False, indent=2)}\n```"
            yield {'type': 'content', 'text': response_text}


def test_full_multiagent_workflow():
    """전체 Multi-Agent 워크플로우 테스트"""
    
    print("=== 전체 Multi-Agent 워크플로우 테스트 ===\n")
    
    try:
        from app.relation_agents import IterativeRelationInferenceAgent
        
        mock_engine = DetailedMockLLMEngine()
        agent = IterativeRelationInferenceAgent(mock_engine)
        
        # 테스트 데이터
        test_text = "딥러닝을 활용한 자연어처리 연구에서 BERT 모델을 사용하여 텍스트 분류 성능을 향상시켰습니다."
        test_entities = [
            {'id': 'entity_1', 'text': '딥러닝', 'type': 'TECHNOLOGY'},
            {'id': 'entity_2', 'text': '자연어처리', 'type': 'FIELD'},
            {'id': 'entity_3', 'text': 'BERT 모델', 'type': 'MODEL'}
        ]
        
        print("1. 테스트 설정:")
        print(f"   - 텍스트: {test_text}")
        print(f"   - 엔티티 수: {len(test_entities)}")
        print(f"   - 엔티티 ID: {[e['id'] for e in test_entities]}")
        
        print("\n2. IterativeRelationInferenceAgent 실행...")
        
        result = agent.infer_relations_iteratively(
            text=test_text,
            entities=test_entities,
            user_context='테스트',
            domain='research_paper',
            additional_guidelines='정확한 관계만 추출',
            max_relations=10
        )
        
        print(f"\n3. 최종 결과:")
        print(f"   - 성공: {result.get('success')}")
        print(f"   - 관계 수: {len(result.get('relations', []))}")
        print(f"   - 최고 점수: {result.get('best_score', 0)}")
        print(f"   - 반복 횟수: {result.get('total_iterations', 0)}")
        
        if result.get('relations'):
            print(f"\n4. 추론된 관계들:")
            for i, relation in enumerate(result['relations']):
                print(f"   관계 {i+1}: {relation.get('subject')} --{relation.get('relation_type')}--> {relation.get('object')} (신뢰도: {relation.get('confidence')})")
        
        if 'error' in result:
            print(f"\n오류: {result['error']}")
            
        print(f"\n5. LLM 호출 통계:")
        print(f"   - 총 호출 수: {mock_engine.call_count}")
        
        return result
        
    except Exception as e:
        logger.error(f"테스트 중 오류: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    result = test_full_multiagent_workflow()
    
    if result and result.get('success') and result.get('relations'):
        print(f"\n✅ 성공: {len(result['relations'])}개 관계가 추론되었습니다!")
    else:
        print(f"\n❌ 실패: 관계 추론에 실패했습니다.")