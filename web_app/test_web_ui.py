#!/usr/bin/env python3
"""
웹 UI에서 엔티티 로딩을 테스트하는 스크립트
"""

import json
import requests

def test_load_sample_data():
    """샘플 데이터 로드 API 테스트"""
    print("🧪 Testing sample data loading...")
    
    try:
        response = requests.get('http://localhost:5001/api/knowledge-graph/load-sample', timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Sample data loaded successfully!")
            
            frames = result.get('frames', [])
            print(f"📋 Loaded {len(frames)} entities:")
            
            for i, frame in enumerate(frames[:5]):  # 처음 5개만 출력
                entity_text = frame.get('entity_text', '')
                entity_type = frame.get('attr', {}).get('entity_type', 'UNKNOWN')
                print(f"   {i+1}. {entity_text} ({entity_type})")
                
            if len(frames) > 5:
                print(f"   ... and {len(frames) - 5} more entities")
                
            return result
        else:
            print(f"❌ Failed to load sample data: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        
    return None

def main():
    """메인 테스트"""
    print("🌐 Testing Web UI Entity Loading")
    print("=" * 50)
    
    # 샘플 데이터 로드 테스트
    sample_data = test_load_sample_data()
    
    if sample_data:
        print("\n✨ Ready for web UI testing!")
        print("1. Open your browser to http://localhost:5001")
        print("2. Navigate to Knowledge Graph section")
        print("3. Load the sample data or upload your own")
        print("4. Generate the knowledge graph")
        print("5. Check if entities are properly displayed")
    else:
        print("\n❌ Sample data loading failed. Check the web server.")

if __name__ == "__main__":
    main()