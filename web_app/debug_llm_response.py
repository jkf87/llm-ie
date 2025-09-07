#!/usr/bin/env python3
"""
LLM 응답 직접 테스트
IterativeRelationInferenceAgent의 파싱 문제를 직접 디버깅
"""

import sys
import os
import json
import logging
import re

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
        
        # 관계 추론용 응답
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
        
        # 파서가 기대하는 형식으로 JSON 블록 포함
        response_text = f"다음은 분석된 관계들입니다:\n\n```json\n{json.dumps(relations_response, ensure_ascii=False, indent=2)}\n```"
        yield {'type': 'content', 'text': response_text}


def test_llm_response_parsing():
    """LLM 응답 파싱 직접 테스트"""
    
    print("=== LLM 응답 파싱 테스트 ===\n")
    
    try:
        from app.relation_agents import IterativeRelationInferenceAgent
        
        mock_engine = MockLLMEngine()
        agent = IterativeRelationInferenceAgent(mock_engine)
        
        # 테스트용 엔티티
        test_entities = [
            {'id': 'entity_1', 'text': '딥러닝', 'type': 'TECHNOLOGY'},
            {'id': 'entity_2', 'text': '자연어처리', 'type': 'FIELD'},
            {'id': 'entity_3', 'text': 'BERT 모델', 'type': 'MODEL'}
        ]
        
        print("1. Mock LLM 응답 생성...")
        
        # LLM 응답 직접 생성
        messages = [{"role": "user", "content": "관계를 추론해주세요"}]
        response_generator = mock_engine.chat_completion(messages)
        
        response_text = ""
        for response in response_generator:
            if response.get('type') == 'content':
                response_text += response.get('text', '')
        
        print(f"2. LLM 응답 텍스트:")
        print(response_text)
        print(f"\n3. 응답 길이: {len(response_text)} 문자")
        
        # 파싱 테스트
        print("\n4. 관계 파싱 테스트...")
        parsed_relations = agent._parse_relations_response(response_text)
        print(f"파싱된 관계 수: {len(parsed_relations)}")
        
        if parsed_relations:
            for i, relation in enumerate(parsed_relations):
                print(f"  관계 {i+1}: {relation}")
        
        # 유효성 검증 테스트
        print("\n5. 유효성 검증 테스트...")
        print(f"엔티티 ID 목록: {[e['id'] for e in test_entities]}")
        
        valid_count = 0
        for i, relation in enumerate(parsed_relations):
            is_valid = (
                isinstance(relation, dict) and
                'subject' in relation and 
                ('relation_type' in relation or 'type' in relation) and 
                'object' in relation and
                relation['subject'] in [e['id'] for e in test_entities] and
                relation['object'] in [e['id'] for e in test_entities]
            )
            print(f"  관계 {i+1} 유효성: {is_valid}")
            if is_valid:
                valid_count += 1
                
        print(f"\n총 유효한 관계: {valid_count}")
        
    except Exception as e:
        logger.error(f"테스트 중 오류: {e}", exc_info=True)


def test_regex_patterns():
    """정규식 패턴 테스트"""
    
    print("\n=== 정규식 패턴 테스트 ===\n")
    
    test_text = """다음은 분석된 관계들입니다:

```json
[
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
```"""

    print("1. 테스트 텍스트:")
    print(test_text)
    
    print("\n2. JSON 블록 찾기 테스트...")
    json_match = re.search(r'```json\s*(\[.*?\])\s*```', test_text, re.DOTALL)
    if json_match:
        json_text = json_match.group(1)
        print("JSON 블록 발견!")
        print(f"추출된 JSON: {json_text[:100]}...")
        
        try:
            relations = json.loads(json_text)
            print(f"파싱 성공! 관계 수: {len(relations)}")
            for relation in relations:
                print(f"  - {relation.get('subject')} --{relation.get('relation_type')}--> {relation.get('object')}")
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {e}")
    else:
        print("JSON 블록을 찾지 못했습니다.")
        
    print("\n3. 일반 JSON 배열 찾기 테스트...")  
    json_match2 = re.search(r'\[.*?\]', test_text, re.DOTALL)
    if json_match2:
        json_text2 = json_match2.group(0)
        print("JSON 배열 발견!")
        print(f"추출된 JSON: {json_text2[:100]}...")


if __name__ == "__main__":
    test_llm_response_parsing()
    test_regex_patterns()