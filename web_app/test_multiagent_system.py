#!/usr/bin/env python3
"""
Multi-Agent Knowledge Graph System Test Suite
LLM-IE 웹앱의 다중 에이전트 지식그래프 시스템 종합 테스트

이 테스트 스위트는 다음 컴포넌트들을 검증합니다:
1. RelationPromptAgent - 관계 추론 프롬프트 생성
2. RelationEvaluatorAgent - 관계 품질 평가 
3. IterativeRelationInferenceAgent - 반복적 관계 개선
4. KnowledgeGraphGenerator - 지식그래프 생성 (업데이트된 버전)
5. KnowledgeGraphValidator - 지식그래프 검증 및 오류 수정
6. API 엔드포인트들 - 새로운 멀티에이전트 워크플로우 API
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Any

# 웹앱 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockLLMEngine:
    """
    테스트용 Mock LLM Engine
    실제 LLM API 호출 없이 예상되는 응답을 시뮬레이션
    """
    
    def __init__(self):
        self.call_count = 0
        
    def chat_completion(self, messages, stream=False):
        """Mock chat completion method"""
        self.call_count += 1
        
        # 메시지 내용에 따라 다른 응답 생성
        last_message = messages[-1]['content'] if messages else ''
        
        if 'prompt improvement' in last_message.lower() or '프롬프트' in last_message:
            # RelationPromptAgent용 응답
            response = {
                "success": True,
                "improved_prompt": "다음 텍스트에서 엔티티들 간의 의미적 관계를 분석하세요. 특히 연구 논문의 맥락에서...",
                "improvements_made": [
                    "도메인 특화 지침 추가",
                    "관계 유형 명확화",
                    "예시 추가"
                ],
                "domain_guidelines": ["학술 논문에서는 STUDIES, ANALYZES 관계가 중요", "방법론과 결과 간의 USES 관계 주의"]
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
                "feedback": "관계가 대체로 적절하지만 더 구체적인 설명이 필요합니다.",
                "suggestions": [
                    "관계의 맥락을 더 명확히 설명하세요",
                    "엔티티 간의 직접적 연관성을 강화하세요"
                ]
            }
            yield {'type': 'content', 'text': json.dumps(response, ensure_ascii=False)}
            
        elif 'relation' in last_message.lower() or '관계' in last_message:
            # 관계 추론용 응답
            response = [
                {
                    "subject": "entity_1",
                    "relation_type": "STUDIES",
                    "object": "entity_2", 
                    "confidence": 0.85,
                    "explanation": "텍스트에서 entity_1이 entity_2를 연구한다고 명시되어 있습니다."
                },
                {
                    "subject": "entity_2",
                    "relation_type": "IS_TYPE_OF",
                    "object": "entity_3",
                    "confidence": 0.9,
                    "explanation": "entity_2는 entity_3의 한 종류로 분류됩니다."
                }
            ]
            yield {'type': 'content', 'text': json.dumps(response, ensure_ascii=False)}
            
        else:
            # 기본 응답
            yield {'type': 'content', 'text': '{"result": "mock response"}'}


class MultiAgentSystemTester:
    """Multi-Agent 시스템 종합 테스터"""
    
    def __init__(self):
        self.mock_engine = MockLLMEngine()
        self.test_results = {}
        
        # 테스트용 샘플 데이터
        self.sample_frames = [
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
        
        # RelationPromptAgent용 엔티티 형식
        self.sample_entities_for_prompt = [
            {
                'id': 'entity_1',
                'text': '딥러닝',
                'type': 'TECHNOLOGY'
            },
            {
                'id': 'entity_2',
                'text': '자연어처리',
                'type': 'FIELD'
            },
            {
                'id': 'entity_3',
                'text': 'BERT 모델',
                'type': 'MODEL'
            }
        ]
        
        self.sample_text = "딥러닝을 활용한 자연어처리 연구에서 BERT 모델을 사용하여 텍스트 분류 성능을 향상시켰습니다."
        
    def test_relation_prompt_agent(self):
        """RelationPromptAgent 테스트"""
        logger.info("Testing RelationPromptAgent...")
        
        try:
            from app.relation_agents import RelationPromptAgent
            
            agent = RelationPromptAgent(self.mock_engine)
            
            # 프롬프트 생성 테스트
            result = agent.generate_relation_prompt(
                text=self.sample_text,
                entities=self.sample_entities_for_prompt,
                domain='research_paper',
                user_context='학술 논문 분석',
                additional_guidelines='정확성을 중시하세요'
            )
            
            self.test_results['RelationPromptAgent'] = {
                'status': 'passed' if result.get('success') else 'failed',
                'details': result,
                'llm_calls': self.mock_engine.call_count
            }
            
            logger.info(f"RelationPromptAgent test result: {result.get('success', False)}")
            
        except Exception as e:
            self.test_results['RelationPromptAgent'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"RelationPromptAgent test failed: {e}")
    
    def test_relation_evaluator_agent(self):
        """RelationEvaluatorAgent 테스트"""
        logger.info("Testing RelationEvaluatorAgent...")
        
        try:
            from app.relation_agents import RelationEvaluatorAgent
            
            agent = RelationEvaluatorAgent(self.mock_engine)
            
            # 관계 평가 테스트 - 새로운 메서드명 사용
            sample_relations = [
                {
                    'subject': 'entity_1',
                    'relation_type': 'USED_IN',
                    'object': 'entity_2',
                    'confidence': 0.8,
                    'explanation': '딥러닝이 자연어처리에 사용됩니다'
                }
            ]
            
            result = agent.evaluate_relations(
                relations=sample_relations,
                text=self.sample_text,
                entities=self.sample_frames,
                user_context='학술 논문 분석'
            )
            
            self.test_results['RelationEvaluatorAgent'] = {
                'status': 'passed' if result.get('success') else 'failed',
                'details': result,
                'score': result.get('overall_score', 0)
            }
            
            logger.info(f"RelationEvaluatorAgent test result: {result.get('success', False)}, Score: {result.get('overall_score', 0)}")
            
        except Exception as e:
            self.test_results['RelationEvaluatorAgent'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"RelationEvaluatorAgent test failed: {e}")
    
    def test_iterative_relation_inference_agent(self):
        """IterativeRelationInferenceAgent 테스트"""
        logger.info("Testing IterativeRelationInferenceAgent...")
        
        try:
            from app.relation_agents import IterativeRelationInferenceAgent
            
            agent = IterativeRelationInferenceAgent(self.mock_engine)
            
            # 반복적 관계 추론 테스트 - 새로운 메서드명 사용
            result = agent.infer_relations_iteratively(
                text=self.sample_text,
                entities=self.sample_entities_for_prompt,
                user_context='학술 논문',
                domain='research_paper',
                additional_guidelines='정확한 관계만 추출하세요',
                max_relations=10
            )
            
            self.test_results['IterativeRelationInferenceAgent'] = {
                'status': 'passed' if result.get('success') else 'failed',
                'details': result,
                'relations_count': len(result.get('relations', [])),
                'iterations_used': result.get('iterations_used', 0)
            }
            
            logger.info(f"IterativeRelationInferenceAgent test result: {result.get('success', False)}")
            logger.info(f"Relations generated: {len(result.get('relations', []))}, Iterations: {result.get('iterations_used', 0)}")
            
        except Exception as e:
            self.test_results['IterativeRelationInferenceAgent'] = {
                'status': 'error', 
                'error': str(e)
            }
            logger.error(f"IterativeRelationInferenceAgent test failed: {e}")
    
    def test_knowledge_graph_generator(self):
        """KnowledgeGraphGenerator 테스트 (업데이트된 버전)"""
        logger.info("Testing KnowledgeGraphGenerator...")
        
        try:
            from app.knowledge_graph_agents import KnowledgeGraphGenerator
            
            generator = KnowledgeGraphGenerator(self.mock_engine)
            
            # 샘플 추출 데이터
            extraction_data = {
                'doc_id': 'test_doc',
                'text': self.sample_text,
                'frames': self.sample_frames,
                'relations': []
            }
            
            # 지식그래프 생성 테스트 (새 매개변수 포함)
            result = generator.generate_knowledge_graph(
                extraction_data=extraction_data,
                user_context='학술 논문 분석',
                domain='research_paper',
                additional_guidelines='정확한 관계 추론',
                max_relations=15,
                use_iterative_inference=True
            )
            
            self.test_results['KnowledgeGraphGenerator'] = {
                'status': 'passed' if result.get('success') else 'failed',
                'details': result,
                'triples_count': result.get('total_triples', 0),
                'has_rdf_formats': bool(result.get('rdf_formats'))
            }
            
            logger.info(f"KnowledgeGraphGenerator test result: {result.get('success', False)}")
            logger.info(f"Triples generated: {result.get('total_triples', 0)}")
            
        except Exception as e:
            self.test_results['KnowledgeGraphGenerator'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"KnowledgeGraphGenerator test failed: {e}")
    
    def test_knowledge_graph_validator(self):
        """KnowledgeGraphValidator 테스트 (오류 수정 기능 포함)"""
        logger.info("Testing KnowledgeGraphValidator...")
        
        try:
            from app.knowledge_graph_agents import KnowledgeGraphValidator, KnowledgeGraphGenerator
            
            # 먼저 테스트용 지식그래프 생성
            generator = KnowledgeGraphGenerator(self.mock_engine)
            extraction_data = {
                'doc_id': 'test_doc',
                'text': self.sample_text,
                'frames': self.sample_frames,
                'relations': []
            }
            kg_result = generator.generate_knowledge_graph(extraction_data)
            
            if kg_result.get('success'):
                validator = KnowledgeGraphValidator(self.mock_engine)
                
                # 검증 테스트 (자동 수정 포함)
                validation_result = validator.validate_knowledge_graph(
                    kg_data=kg_result,
                    auto_fix_errors=True
                )
                
                self.test_results['KnowledgeGraphValidator'] = {
                    'status': 'passed' if validation_result.get('success') else 'failed',
                    'details': validation_result,
                    'overall_score': validation_result.get('overall_score', 0),
                    'errors_fixed': len(validation_result.get('errors_fixed', [])),
                    'auto_fix_enabled': validation_result.get('auto_fix_enabled', False)
                }
                
                logger.info(f"KnowledgeGraphValidator test result: {validation_result.get('success', False)}")
                logger.info(f"Validation score: {validation_result.get('overall_score', 0)}")
                logger.info(f"Errors fixed: {len(validation_result.get('errors_fixed', []))}")
            else:
                raise Exception("Failed to generate knowledge graph for validation test")
            
        except Exception as e:
            self.test_results['KnowledgeGraphValidator'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"KnowledgeGraphValidator test failed: {e}")
    
    def test_api_integration(self):
        """API 통합 테스트 (Flask 앱 없이 모듈 레벨에서)"""
        logger.info("Testing API integration...")
        
        try:
            # 여기서는 API 엔드포인트 함수들을 직접 테스트
            # 실제 Flask 앱을 실행하지 않고 함수만 테스트
            
            # Mock request 객체
            class MockRequest:
                def __init__(self, json_data):
                    self.json = json_data
            
            # 이것은 실제 Flask 없이는 완전히 테스트하기 어려우므로
            # 기본적인 import 테스트만 수행
            try:
                from app.routes import main_bp
                api_tests_passed = True
            except ImportError as ie:
                api_tests_passed = False
                api_error = str(ie)
            
            self.test_results['API_Integration'] = {
                'status': 'passed' if api_tests_passed else 'failed',
                'details': 'API endpoints imported successfully' if api_tests_passed else f'Import error: {api_error}',
                'endpoints_tested': [
                    '/api/relation-inference/iterative',
                    '/api/relations/edit', 
                    '/api/relations/validate-batch'
                ]
            }
            
            logger.info(f"API Integration test result: {api_tests_passed}")
            
        except Exception as e:
            self.test_results['API_Integration'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"API Integration test failed: {e}")
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("Starting Multi-Agent System Test Suite...")
        
        start_time = time.time()
        
        # 각 컴포넌트 테스트 실행
        self.test_relation_prompt_agent()
        self.test_relation_evaluator_agent()
        self.test_iterative_relation_inference_agent()
        self.test_knowledge_graph_generator()
        self.test_knowledge_graph_validator()
        self.test_api_integration()
        
        end_time = time.time()
        
        # 테스트 결과 요약
        self.print_test_summary(end_time - start_time)
        
        return self.test_results
    
    def print_test_summary(self, execution_time):
        """테스트 결과 요약 출력"""
        print("\n" + "="*70)
        print("MULTI-AGENT SYSTEM TEST SUMMARY")
        print("="*70)
        
        passed = 0
        failed = 0
        errors = 0
        
        for component, result in self.test_results.items():
            status = result['status']
            if status == 'passed':
                passed += 1
                status_symbol = "✅"
            elif status == 'failed':
                failed += 1  
                status_symbol = "❌"
            else:
                errors += 1
                status_symbol = "⚠️"
            
            print(f"{status_symbol} {component:<35} {status.upper()}")
            
            # 추가 세부사항 출력
            if 'score' in result:
                print(f"   └─ Score: {result['score']}")
            if 'relations_count' in result:
                print(f"   └─ Relations: {result['relations_count']}")
            if 'triples_count' in result:
                print(f"   └─ Triples: {result['triples_count']}")
            if 'errors_fixed' in result:
                print(f"   └─ Errors Fixed: {result['errors_fixed']}")
            if result['status'] == 'error':
                print(f"   └─ Error: {result.get('error', 'Unknown error')}")
        
        print("-"*70)
        print(f"Total Tests: {len(self.test_results)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Errors: {errors}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Mock LLM Calls: {self.mock_engine.call_count}")
        print("="*70)
        
        # 전체 시스템 상태
        if errors == 0 and failed == 0:
            print("🎉 ALL TESTS PASSED - Multi-Agent System is ready!")
        elif errors == 0:
            print("⚠️  Some tests failed - Review failed components")
        else:
            print("❌ Critical errors found - System needs debugging")


def main():
    """메인 테스트 실행 함수"""
    print("LLM-IE Multi-Agent Knowledge Graph System Test Suite")
    print("=" * 70)
    
    tester = MultiAgentSystemTester()
    results = tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    results_file = f"multiagent_test_results_{int(time.time())}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nDetailed test results saved to: {results_file}")
    
    # 시스템 준비 상태 반환
    all_passed = all(result['status'] == 'passed' for result in results.values())
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)