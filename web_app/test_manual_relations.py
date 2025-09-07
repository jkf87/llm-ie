#!/usr/bin/env python3
"""
수동 관계 생성 기능 테스트 스크립트
"""

import json
import requests

def test_manual_relation_api():
    """수동 관계 생성 API 테스트"""
    print("🧪 Testing manual relation API...")
    
    # 테스트 관계 데이터
    test_relation = {
        'from': 'entity_1',
        'to': 'entity_2', 
        'label': '포함하다',
        'description': '첫 번째 엔티티가 두 번째 엔티티를 포함합니다',
        'confidence': 0.9,
        'direction': 'directed'
    }
    
    try:
        response = requests.post(
            'http://localhost:5001/api/knowledge-graph/manual-relation',
            json={'relation': test_relation},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Manual relation API test passed!")
            print(f"   Relation ID: {result.get('relation', {}).get('id')}")
            print(f"   Label: {result.get('relation', {}).get('label')}")
            print(f"   Confidence: {result.get('relation', {}).get('confidence')}")
            return True
        else:
            print(f"❌ API test failed: HTTP {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

def test_export_with_manual_relations():
    """수동 관계 포함 내보내기 API 테스트"""
    print("\n🧪 Testing export with manual relations...")
    
    # 테스트 지식그래프 데이터
    kg_data = {
        'doc_id': 'test_doc',
        'relations': [
            {'from': 'A', 'to': 'B', 'label': 'original_relation'}
        ],
        'statistics': {
            'total_relations': 1
        }
    }
    
    # 테스트 수동 관계들
    manual_relations = [
        {'from': 'A', 'to': 'C', 'label': 'manual_relation_1', 'manual': True},
        {'from': 'B', 'to': 'C', 'label': 'manual_relation_2', 'manual': True}
    ]
    
    try:
        response = requests.post(
            'http://localhost:5001/api/knowledge-graph/export-with-manual-relations',
            json={
                'kg_data': kg_data,
                'manual_relations': manual_relations,
                'format': 'json'
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            exported_data = result.get('data', {})
            
            print("✅ Export API test passed!")
            print(f"   Total relations: {len(exported_data.get('relations', []))}")
            print(f"   Manual relations: {exported_data.get('statistics', {}).get('manual_relations', 0)}")
            print(f"   Export timestamp: {exported_data.get('export_info', {}).get('exported_at')}")
            
            # 관계 내용 확인
            all_relations = exported_data.get('relations', [])
            print(f"   Relations preview:")
            for i, rel in enumerate(all_relations[:3]):
                print(f"     {i+1}. {rel.get('from')} -> {rel.get('to')} ({rel.get('label')})")
            
            return True
        else:
            print(f"❌ Export test failed: HTTP {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Export test error: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 Manual Relationship Creation Tests")
    print("=" * 50)
    
    # API 테스트
    api_success = test_manual_relation_api()
    export_success = test_export_with_manual_relations()
    
    print("\n" + "=" * 50)
    print("📋 Test Results:")
    print(f"   Manual Relation API: {'✅' if api_success else '❌'}")
    print(f"   Export with Manual Relations: {'✅' if export_success else '❌'}")
    
    if api_success and export_success:
        print("\n🎉 All tests passed! Manual relationship creation is working.")
        print("\n📖 How to use manual relationship creation:")
        print("1. Open the knowledge graph in your browser (http://localhost:5001)")
        print("2. Generate a knowledge graph from your data")
        print("3. Click '🔗 수동 관계 생성' button")
        print("4. Click on the first node")
        print("5. Hold Ctrl and click on the second node") 
        print("6. Fill in the relationship form")
        print("7. Click '관계 생성' to add the relationship")
    else:
        print("\n❌ Some tests failed. Check the server logs for details.")

if __name__ == "__main__":
    main()