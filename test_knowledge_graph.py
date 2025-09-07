#!/usr/bin/env python3
"""
지식그래프 기능 테스트 스크립트
Knowledge Graph functionality test script
"""

import json
import sys
import os
from pathlib import Path

# 웹 앱 경로 추가
web_app_path = Path(__file__).parent / "web_app"
sys.path.insert(0, str(web_app_path))

from app.knowledge_graph_agents import KnowledgeGraphGenerator, KnowledgeGraphValidator

def test_knowledge_graph_generation():
    """지식그래프 생성 테스트"""
    print("🧪 Testing Knowledge Graph Generation...")
    
    # 테스트 데이터
    test_data = {
        "doc_id": "test_document",
        "text": "이 논문은 Llama2, GPT4, Mixtral과 같은 LLM들이 어떠한 \"성격\"을 시뮬레이션하는지, 그리고 그 성격이 프롬프트나 온도(temperature) 설정에 따라 얼마나 안정적인지를 IPIP-NEO-120 설문지를 통해 분석한 연구입니다.",
        "frames": [
            {
                "frame_id": "0",
                "start": 5,
                "end": 7,
                "entity_text": "논문",
                "attr": {"entity_type": "DOCUMENT"}
            },
            {
                "frame_id": "1",
                "start": 9,
                "end": 15,
                "entity_text": "Llama2",
                "attr": {"entity_type": "MODEL"}
            },
            {
                "frame_id": "2",
                "start": 17,
                "end": 21,
                "entity_text": "GPT4",
                "attr": {"entity_type": "MODEL"}
            },
            {
                "frame_id": "3",
                "start": 23,
                "end": 30,
                "entity_text": "Mixtral",
                "attr": {"entity_type": "MODEL"}
            },
            {
                "frame_id": "4",
                "start": 35,
                "end": 38,
                "entity_text": "LLM",
                "attr": {"entity_type": "TECHNOLOGY"}
            }
        ],
        "relations": []
    }
    
    # 지식그래프 생성기 초기화 (LLM 없이)
    kg_generator = KnowledgeGraphGenerator(llm_engine=None)
    
    try:
        # 지식그래프 생성
        result = kg_generator.generate_knowledge_graph(test_data)
        
        if result['success']:
            print("✅ Knowledge graph generation successful!")
            print(f"   📊 Total triples: {result['total_triples']}")
            print(f"   📄 Statistics: {result['statistics']}")
            print(f"   🎯 Nodes: {len(result['visualization_data']['nodes'])}")
            print(f"   🔗 Edges: {len(result['visualization_data']['edges'])}")
            
            # RDF 출력 샘플
            turtle_preview = result['rdf_formats']['turtle'][:500]
            print(f"\n📋 RDF (Turtle) Preview:")
            print(turtle_preview + "..." if len(turtle_preview) == 500 else turtle_preview)
            
            return result
        else:
            print(f"❌ Knowledge graph generation failed: {result['error']}")
            return None
            
    except Exception as e:
        print(f"🚨 Test failed with exception: {e}")
        return None

def test_knowledge_graph_validation(kg_data):
    """지식그래프 검증 테스트"""
    if not kg_data:
        print("⏭️  Skipping validation test (no KG data)")
        return
    
    print("\n🔍 Testing Knowledge Graph Validation...")
    
    # 지식그래프 검증기 초기화 (LLM 없이)
    kg_validator = KnowledgeGraphValidator(llm_engine=None)
    
    try:
        # 지식그래프 검증
        result = kg_validator.validate_knowledge_graph(kg_data)
        
        if result['success']:
            print("✅ Knowledge graph validation successful!")
            print(f"   🎯 Overall score: {result['overall_score']:.1f}/100")
            print(f"   ✔️  Passed checks: {result['passed_checks']}/{result['total_checks']}")
            
            # 검증 결과 요약
            print("\n📋 Validation Results:")
            for check in result['validation_results']:
                status = "✅" if check['passed'] else "❌"
                print(f"   {status} {check['rule']}: {check['message']}")
            
            # 추천사항
            if result['recommendations']:
                print(f"\n💡 Recommendations:")
                for rec in result['recommendations']:
                    print(f"   • {rec}")
            
            return result
        else:
            print(f"❌ Knowledge graph validation failed: {result['error']}")
            return None
            
    except Exception as e:
        print(f"🚨 Validation test failed with exception: {e}")
        return None

def test_file_based_generation():
    """파일 기반 지식그래프 생성 테스트"""
    print("\n📁 Testing File-based Knowledge Graph Generation...")
    
    # 실제 파일 로드
    extraction_file = Path(__file__).parent / "extraction_results_20250902-173510.llmie"
    
    if not extraction_file.exists():
        print("⏭️  Skipping file-based test (extraction file not found)")
        return None
    
    try:
        with open(extraction_file, 'r', encoding='utf-8') as f:
            extraction_data = json.load(f)
        
        print(f"📄 Loaded extraction data: {extraction_data['doc_id']}")
        print(f"   📊 Frames: {len(extraction_data.get('frames', []))}")
        print(f"   🔗 Relations: {len(extraction_data.get('relations', []))}")
        
        # 지식그래프 생성
        kg_generator = KnowledgeGraphGenerator(llm_engine=None)
        result = kg_generator.generate_knowledge_graph(extraction_data)
        
        if result['success']:
            print("✅ File-based knowledge graph generation successful!")
            print(f"   📊 Total triples: {result['total_triples']}")
            print(f"   🎯 Entity types: {result['statistics']['entity_types']}")
            
            # 생성된 지식그래프를 파일로 저장
            output_file = Path(__file__).parent / "test_knowledge_graph_output.ttl"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['rdf_formats']['turtle'])
            print(f"💾 RDF saved to: {output_file}")
            
            return result
        else:
            print(f"❌ File-based generation failed: {result['error']}")
            return None
            
    except Exception as e:
        print(f"🚨 File-based test failed with exception: {e}")
        return None

def main():
    """메인 테스트 함수"""
    print("🚀 Starting Knowledge Graph Tests")
    print("=" * 50)
    
    # 기본 생성 테스트
    kg_data = test_knowledge_graph_generation()
    
    # 검증 테스트
    test_knowledge_graph_validation(kg_data)
    
    # 파일 기반 테스트
    file_kg_data = test_file_based_generation()
    if file_kg_data:
        test_knowledge_graph_validation(file_kg_data)
    
    print("\n" + "=" * 50)
    print("🎉 Knowledge Graph Tests Completed!")

if __name__ == "__main__":
    main()