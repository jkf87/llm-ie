#!/usr/bin/env python3
"""
엔티티 로딩 문제 디버깅 스크립트
"""

import json
import requests
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_knowledge_graph_generation():
    """지식그래프 생성 API 테스트"""
    
    # 테스트 데이터 로드
    with open('/Users/conanssam-m4/llm-ie/extraction_results_20250902-173510.llmie', 'r') as f:
        extraction_data = json.load(f)
    
    logger.info(f"Loaded extraction data with {len(extraction_data.get('frames', []))} entities")
    
    # API 요청 데이터 구성
    request_data = {
        "extraction_data": extraction_data,
        "settings": {
            "enable_llm_inference": False,  # LLM 없이 테스트
            "user_context": "한국어 논문 분석",
            "domain": "research_paper",
            "max_relations": 10,
            "use_iterative_inference": False,  # 간단한 테스트를 위해 비활성화
            "enable_entity_resolution": True  # 엔티티 해결 활성화
        }
    }
    
    # API 호출
    try:
        logger.info("Calling knowledge graph generation API...")
        response = requests.post(
            'http://localhost:5001/api/knowledge-graph/generate',
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                logger.info("✅ Knowledge graph generation successful!")
                
                # 통계 출력
                stats = result.get('statistics', {})
                logger.info(f"📊 Statistics:")
                logger.info(f"   Total entities: {stats.get('total_entities', 0)}")
                logger.info(f"   Total relations: {stats.get('total_relations', 0)}")
                logger.info(f"   Total triples: {result.get('total_triples', 0)}")
                
                # 엔티티 타입별 통계
                entity_types = stats.get('entity_types', {})
                if entity_types:
                    logger.info(f"   Entity types: {entity_types}")
                
                # 시각화 데이터 확인
                viz_data = result.get('visualization_data', {})
                nodes = viz_data.get('nodes', [])
                edges = viz_data.get('edges', [])
                
                logger.info(f"🎨 Visualization data:")
                logger.info(f"   Nodes: {len(nodes)}")
                logger.info(f"   Edges: {len(edges)}")
                
                # 처음 몇 개 노드 출력
                if nodes:
                    logger.info(f"   First few nodes:")
                    for node in nodes[:5]:
                        logger.info(f"     - {node.get('id')}: {node.get('label')} ({node.get('group', 'unknown')})")
                else:
                    logger.error("   ❌ No nodes found in visualization data!")
                
                return result
                
            else:
                logger.error(f"❌ API returned error: {result.get('error')}")
                return None
                
        else:
            logger.error(f"❌ HTTP error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Exception during API call: {e}")
        return None

def test_without_entity_resolution():
    """엔티티 해결 없이 테스트"""
    
    # 테스트 데이터 로드
    with open('/Users/conanssam-m4/llm-ie/extraction_results_20250902-173510.llmie', 'r') as f:
        extraction_data = json.load(f)
    
    logger.info("Testing WITHOUT entity resolution...")
    
    request_data = {
        "extraction_data": extraction_data,
        "settings": {
            "enable_llm_inference": False,
            "user_context": "",
            "domain": "general",
            "max_relations": 10,
            "use_iterative_inference": False,
            "enable_entity_resolution": False  # 엔티티 해결 비활성화
        }
    }
    
    try:
        response = requests.post(
            'http://localhost:5001/api/knowledge-graph/generate',
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                stats = result.get('statistics', {})
                logger.info(f"✅ Without entity resolution - Total entities: {stats.get('total_entities', 0)}")
                return result
            else:
                logger.error(f"❌ Error without entity resolution: {result.get('error')}")
        else:
            logger.error(f"❌ HTTP error without entity resolution: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Exception without entity resolution: {e}")
        
    return None

def main():
    """메인 디버깅 실행"""
    logger.info("🔍 Starting entity loading debug...")
    
    # 1. 엔티티 해결 비활성화 테스트
    result_without = test_without_entity_resolution()
    
    # 2. 엔티티 해결 활성화 테스트
    result_with = test_knowledge_graph_generation()
    
    # 3. 비교
    if result_without and result_with:
        entities_without = result_without.get('statistics', {}).get('total_entities', 0)
        entities_with = result_with.get('statistics', {}).get('total_entities', 0)
        
        logger.info(f"🔄 Comparison:")
        logger.info(f"   Without entity resolution: {entities_without} entities")
        logger.info(f"   With entity resolution: {entities_with} entities")
        
        if entities_without > 0 and entities_with == 0:
            logger.error("❌ Entity resolution is causing entities to disappear!")
        elif entities_without == 0 and entities_with == 0:
            logger.error("❌ No entities are being loaded at all!")
        else:
            logger.info("✅ Entity loading appears to be working")

if __name__ == "__main__":
    main()