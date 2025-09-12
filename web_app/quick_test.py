#!/usr/bin/env python3
import json
import requests

# 데이터 로드
with open('/Users/conanssam-m4/llm-ie/extraction_results_20250902-173510.llmie', 'r') as f:
    extraction_data = json.load(f)

# API 요청
request_data = {
    "extraction_data": extraction_data,
    "settings": {"enable_entity_resolution": False}
}

response = requests.post('http://localhost:5001/api/knowledge-graph/generate', json=request_data)
result = response.json()

print(f"Entities: {result.get('statistics', {}).get('total_entities', 0)}")
print(f"Entity types: {result.get('statistics', {}).get('entity_types', {})}")