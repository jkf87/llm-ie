#!/usr/bin/env python3
import json
import requests

# 데이터 로드
with open('/Users/conanssam-m4/llm-ie/extraction_results_20250902-173510.llmie', 'r') as f:
    extraction_data = json.load(f)

print("🔍 Testing Entity Resolution...")
print(f"Original entities: {len(extraction_data.get('frames', []))}")

# API 요청 (엔티티 해결 활성화)
request_data = {
    "extraction_data": extraction_data,
    "settings": {"enable_entity_resolution": True}
}

response = requests.post('http://localhost:5001/api/knowledge-graph/generate', json=request_data)
result = response.json()

print(f"✅ Resolved entities: {result.get('statistics', {}).get('total_entities', 0)}")
print(f"Entity types: {result.get('statistics', {}).get('entity_types', {})}")

# 시각화 데이터 확인
viz_data = result.get('visualization_data', {})
nodes = viz_data.get('nodes', [])

print(f"\n📊 Visualization nodes ({len(nodes)}):")
for node in nodes:
    print(f"  - {node.get('label', 'N/A')} ({node.get('group', 'N/A')})")
    
print(f"\n🎯 Entity resolution reduced entities from {len(extraction_data.get('frames', []))} to {len(nodes)}")

if len(nodes) < len(extraction_data.get('frames', [])):
    print("✅ Entity resolution is working correctly!")
else:
    print("⚠️ No entity reduction occurred")