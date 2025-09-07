"""
Knowledge Graph Sub-agents for LLM-IE Web Application
지식그래프 생성 및 검증을 위한 서브 에이전트들
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal, BNode
from rdflib.namespace import XSD, FOAF
import re
from datetime import datetime

# 관계 추론 에이전트들 import
try:
    from .relation_agents import IterativeRelationInferenceAgent
except ImportError:
    # 테스트 환경에서는 상대 import가 실패할 수 있음
    try:
        from relation_agents import IterativeRelationInferenceAgent
    except ImportError:
        IterativeRelationInferenceAgent = None
        logger.warning("IterativeRelationInferenceAgent not available")


logger = logging.getLogger(__name__)


class KnowledgeGraphGenerator:
    """
    지식그래프 생성을 위한 Sub-agent
    LLM-IE 추출 결과를 RDF 형태의 지식그래프로 변환
    """
    
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine
        
        # RDF 네임스페이스 정의
        self.LLMIE = Namespace("http://llm-ie.org/ontology#")
        self.ENTITY = Namespace("http://llm-ie.org/entity#")
        self.RELATION = Namespace("http://llm-ie.org/relation#")
        self.DOC = Namespace("http://llm-ie.org/document#")
        
        # RDF 그래프 초기화
        self.graph = Graph()
        self.graph.bind("llmie", self.LLMIE)
        self.graph.bind("entity", self.ENTITY)
        self.graph.bind("relation", self.RELATION)
        self.graph.bind("doc", self.DOC)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("foaf", FOAF)
    
    def generate_knowledge_graph(self, extraction_data: Dict, 
                                user_context: str = "",
                                domain: str = "general", 
                                additional_guidelines: str = "",
                                max_relations: int = 20,
                                use_iterative_inference: bool = True) -> Dict[str, Any]:
        """
        LLM-IE 추출 결과로부터 지식그래프 생성 (Multi-Agent 시스템 사용)
        
        Args:
            extraction_data: LLM-IE 추출 결과 딕셔너리
            user_context: 사용자 제공 컨텍스트
            domain: 문서 도메인 (research_paper, technical_document, business_document, general)
            additional_guidelines: 추가 가이드라인
            max_relations: 최대 관계 수
            use_iterative_inference: 반복적 개선 사용 여부
            
        Returns:
            지식그래프 생성 결과
        """
        try:
            # 새로운 그래프 초기화
            self.graph = Graph()
            self._bind_namespaces()
            
            doc_id = extraction_data.get('doc_id', 'unknown_document')
            text = extraction_data.get('text', '')
            frames = extraction_data.get('frames', [])
            relations = extraction_data.get('relations', [])
            
            logger.info(f"Generating knowledge graph for document: {doc_id}")
            logger.info(f"Processing {len(frames)} entities and {len(relations)} relations")
            
            # 문서 노드 생성
            doc_uri = self._create_document_node(doc_id, text)
            
            # 엔티티 노드들 생성
            entity_uris = {}
            for frame in frames:
                entity_uri = self._create_entity_node(frame, doc_uri)
                entity_uris[frame['frame_id']] = entity_uri
            
            # 관계 생성 - Multi-Agent 시스템 시도하지만 기본 LLM으로 폴백 보장
            if self.llm_engine and use_iterative_inference and IterativeRelationInferenceAgent:
                try:
                    # Iterative Relation Inference Agent 초기화
                    relation_agent = IterativeRelationInferenceAgent(self.llm_engine)
                    
                    # 관계 추론 실행 - 올바른 메서드명과 매개변수 사용
                    # 먼저 entities를 올바른 형식으로 변환
                    entities_for_inference = []
                    for frame in frames:
                        entities_for_inference.append({
                            'id': frame.get('frame_id'),
                            'text': frame.get('entity_text'),
                            'type': frame.get('attr', {}).get('entity_type', 'UNKNOWN')
                        })
                    
                    inference_result = relation_agent.infer_relations_iteratively(
                        text=text,
                        entities=entities_for_inference,
                        user_context=user_context,
                        domain=domain,
                        additional_guidelines=additional_guidelines,
                        max_relations=max_relations
                    )
                    
                    if inference_result.get('success'):
                        inferred_relations = inference_result.get('relations', [])
                        logger.info(f"Multi-agent inference result keys: {list(inference_result.keys())}")
                        logger.info(f"Multi-agent system inferred {len(inferred_relations)} relations")
                        logger.debug(f"Relations: {inferred_relations[:3]}...")  # 첫 3개만 로그
                        relations.extend(inferred_relations)
                    else:
                        logger.warning(f"Multi-agent relation inference failed: {inference_result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Error with multi-agent relation inference: {e}")
            
            # 기본 LLM 추론 - Multi-agent 성공 여부와 관계없이 항상 실행하여 관계 보장  
            if self.llm_engine and frames and len(relations) == 0:  # 관계가 없는 경우에만 기본 LLM 실행
                logger.info("No relations from multi-agent system, using basic LLM inference as fallback")
                fallback_relations = self._infer_relations_with_llm(frames, text)
                relations.extend(fallback_relations)
            
            # 관계 노드들 생성
            logger.info(f"Creating {len(relations)} relations with {len(entity_uris)} entities")
            for i, relation in enumerate(relations):
                logger.debug(f"Processing relation {i+1}: {relation}")
                self._create_relation_node(relation, entity_uris)
            
            # RDF 직렬화
            rdf_formats = {
                'turtle': self.graph.serialize(format='turtle'),
                'xml': self.graph.serialize(format='xml'),
                'json-ld': self.graph.serialize(format='json-ld'),
                'n3': self.graph.serialize(format='n3')
            }
            
            # 통계 생성
            stats = self._generate_statistics()
            
            # 시각화를 위한 노드-엣지 데이터 생성
            viz_data = self._generate_visualization_data()
            
            return {
                'success': True,
                'doc_id': doc_id,
                'rdf_formats': rdf_formats,
                'statistics': stats,
                'visualization_data': viz_data,
                'timestamp': datetime.now().isoformat(),
                'total_triples': len(self.graph)
            }
            
        except Exception as e:
            logger.error(f"Error generating knowledge graph: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _bind_namespaces(self):
        """네임스페이스 바인딩"""
        self.graph.bind("llmie", self.LLMIE)
        self.graph.bind("entity", self.ENTITY)
        self.graph.bind("relation", self.RELATION)
        self.graph.bind("doc", self.DOC)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("foaf", FOAF)
    
    def _create_document_node(self, doc_id: str, text: str) -> URIRef:
        """문서 노드 생성"""
        doc_uri = self.DOC[self._sanitize_uri(doc_id)]
        
        self.graph.add((doc_uri, RDF.type, self.LLMIE.Document))
        self.graph.add((doc_uri, RDFS.label, Literal(doc_id)))
        self.graph.add((doc_uri, self.LLMIE.hasText, Literal(text, lang='ko')))
        self.graph.add((doc_uri, self.LLMIE.createdAt, Literal(datetime.now(), datatype=XSD.dateTime)))
        
        return doc_uri
    
    def _create_entity_node(self, frame: Dict, doc_uri: URIRef) -> URIRef:
        """엔티티 노드 생성"""
        entity_text = frame.get('entity_text', '')
        frame_id = frame.get('frame_id', '')
        start = frame.get('start', 0)
        end = frame.get('end', 0)
        attributes = frame.get('attr', {})
        
        # 엔티티 URI 생성
        entity_uri = self.ENTITY[self._sanitize_uri(f"{frame_id}_{entity_text}")]
        
        # 엔티티 타입 결정
        entity_type = attributes.get('entity_type', 'UNKNOWN')
        entity_class = self.LLMIE[entity_type]
        
        # 기본 트리플 추가
        self.graph.add((entity_uri, RDF.type, entity_class))
        self.graph.add((entity_uri, RDF.type, self.LLMIE.Entity))
        self.graph.add((entity_uri, RDFS.label, Literal(entity_text, lang='ko')))
        self.graph.add((entity_uri, self.LLMIE.hasFrameId, Literal(frame_id)))
        self.graph.add((entity_uri, self.LLMIE.startPosition, Literal(start, datatype=XSD.integer)))
        self.graph.add((entity_uri, self.LLMIE.endPosition, Literal(end, datatype=XSD.integer)))
        self.graph.add((entity_uri, self.LLMIE.extractedFrom, doc_uri))
        
        # 속성들 추가
        for attr_name, attr_value in attributes.items():
            if attr_name != 'entity_type' and attr_value:
                attr_predicate = self.LLMIE[f"has{attr_name.title()}"]
                self.graph.add((entity_uri, attr_predicate, Literal(str(attr_value))))
        
        return entity_uri
    
    def _create_relation_node(self, relation: Dict, entity_uris: Dict[str, URIRef]):
        """관계 노드 생성"""
        # 다양한 관계 데이터 형식 지원
        relation_type = relation.get('relation_type') or relation.get('type', 'RELATED_TO')
        subject_id = relation.get('subject', '')
        object_id = relation.get('object', '')
        confidence = relation.get('confidence', 1.0)
        
        logger.debug(f"Creating relation: {subject_id} --{relation_type}--> {object_id}")
        logger.debug(f"Available entity URIs: {list(entity_uris.keys())}")
        
        if subject_id in entity_uris and object_id in entity_uris:
            subject_uri = entity_uris[subject_id]
            object_uri = entity_uris[object_id]
            
            # 관계 프로퍼티 생성
            relation_predicate = self.RELATION[relation_type]
            
            # 관계 트리플 추가
            self.graph.add((subject_uri, relation_predicate, object_uri))
            logger.info(f"Added relation: {subject_id} --{relation_type}--> {object_id}")
            
            # 관계에 대한 메타데이터 (reification 사용)
            relation_node = BNode()
            self.graph.add((relation_node, RDF.type, RDF.Statement))
            self.graph.add((relation_node, RDF.subject, subject_uri))
            self.graph.add((relation_node, RDF.predicate, relation_predicate))
            self.graph.add((relation_node, RDF.object, object_uri))
            self.graph.add((relation_node, self.LLMIE.hasConfidence, Literal(confidence, datatype=XSD.float)))
        else:
            logger.warning(f"Skipping relation - missing entities: subject='{subject_id}' in URIs: {subject_id in entity_uris}, object='{object_id}' in URIs: {object_id in entity_uris}")
    
    def _infer_relations_with_llm(self, frames: List[Dict], text: str) -> List[Dict]:
        """LLM을 사용하여 암시적 관계 추론"""
        if not self.llm_engine:
            return []
        
        try:
            # 엔티티 목록 생성
            entities = []
            for frame in frames:
                entities.append({
                    'id': frame['frame_id'],
                    'text': frame['entity_text'],
                    'type': frame.get('attr', {}).get('entity_type', 'UNKNOWN')
                })
            
            # LLM에게 관계 추론 요청
            prompt = f"""
다음 텍스트에서 추출된 엔티티들 사이의 관계를 분석해주세요.

텍스트: {text}

엔티티들:
{json.dumps(entities, ensure_ascii=False, indent=2)}

엔티티들 사이의 의미적 관계를 찾아서 다음 형식으로 반환해주세요:
[
    {{
        "subject": "entity_id1",
        "relation_type": "RELATION_NAME",
        "object": "entity_id2",
        "confidence": 0.8,
        "explanation": "관계 설명"
    }}
]

가능한 관계 타입:
- PART_OF: A가 B의 일부
- USES: A가 B를 사용
- STUDIES: A가 B를 연구
- MEASURES: A가 B를 측정
- DESCRIBES: A가 B를 설명
- ANALYZES: A가 B를 분석
- IMPLEMENTS: A가 B를 구현
- RELATED_TO: 일반적인 관련성

JSON 형식으로만 응답해주세요.
"""
            
            messages = [{"role": "user", "content": prompt}]
            
            # LLM 응답 받기
            response_generator = self.llm_engine.chat_completion(messages, stream=False)
            response_text = ""
            
            for response in response_generator:
                if response.get('type') == 'content':
                    response_text += response.get('text', '')
            
            # JSON 파싱
            try:
                relations = json.loads(response_text.strip())
                logger.info(f"LLM inferred {len(relations)} relations")
                return relations
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM relation inference response")
                return []
                
        except Exception as e:
            logger.error(f"Error inferring relations with LLM: {e}")
            return []
    
    def _generate_statistics(self) -> Dict[str, int]:
        """지식그래프 통계 생성"""
        stats = {
            'total_triples': len(self.graph),
            'total_entities': len(list(self.graph.subjects(RDF.type, self.LLMIE.Entity))),
            'total_documents': len(list(self.graph.subjects(RDF.type, self.LLMIE.Document))),
            'total_relations': len(list(self.graph.subjects(RDF.type, RDF.Statement))),
            'entity_types': {}
        }
        
        # 엔티티 타입별 통계
        for entity in self.graph.subjects(RDF.type, self.LLMIE.Entity):
            for entity_type in self.graph.objects(entity, RDF.type):
                if entity_type != self.LLMIE.Entity:
                    type_name = str(entity_type).split('#')[-1]
                    stats['entity_types'][type_name] = stats['entity_types'].get(type_name, 0) + 1
        
        return stats
    
    def _generate_visualization_data(self) -> Dict[str, List]:
        """시각화를 위한 노드-엣지 데이터 생성"""
        nodes = []
        edges = []
        
        # 엔티티 노드들
        for entity in self.graph.subjects(RDF.type, self.LLMIE.Entity):
            label = str(list(self.graph.objects(entity, RDFS.label))[0])
            entity_types = [str(t).split('#')[-1] for t in self.graph.objects(entity, RDF.type) 
                           if t != self.LLMIE.Entity]
            entity_type = entity_types[0] if entity_types else 'UNKNOWN'
            
            nodes.append({
                'id': str(entity),
                'label': label,
                'type': entity_type,
                'group': entity_type
            })
        
        # 관계 엣지들
        for stmt in self.graph.subjects(RDF.type, RDF.Statement):
            subject = list(self.graph.objects(stmt, RDF.subject))[0]
            predicate = list(self.graph.objects(stmt, RDF.predicate))[0]
            obj = list(self.graph.objects(stmt, RDF.object))[0]
            
            confidence_list = list(self.graph.objects(stmt, self.LLMIE.hasConfidence))
            confidence = float(confidence_list[0]) if confidence_list else 1.0
            
            edges.append({
                'from': str(subject),
                'to': str(obj),
                'label': str(predicate).split('#')[-1] if '#' in str(predicate) else str(predicate).split('/')[-1],
                'confidence': confidence
            })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def _sanitize_uri(self, text: str) -> str:
        """URI에 사용 가능한 문자열로 변환"""
        # 한글과 특수문자를 제거하고 언더스코어로 대체
        sanitized = re.sub(r'[^\w\-_.]', '_', text)
        # 연속된 언더스코어 제거
        sanitized = re.sub(r'_+', '_', sanitized)
        # 시작과 끝의 언더스코어 제거
        return sanitized.strip('_')


class KnowledgeGraphValidator:
    """
    지식그래프 검증 및 오류 수정을 위한 Enhanced Sub-agent
    생성된 지식그래프의 품질과 일관성을 검사하고 오류를 자동 수정
    """
    
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine
        self.validation_rules = [
            self._check_basic_structure,
            self._check_entity_consistency,
            self._check_relation_validity,
            self._check_namespace_usage,
            self._check_data_quality
        ]
        self.error_fixing_rules = {
            'basic_structure': self._fix_basic_structure_errors,
            'entity_consistency': self._fix_entity_consistency_errors,
            'relation_validity': self._fix_relation_validity_errors,
            'namespace_usage': self._fix_namespace_usage_errors,
            'data_quality': self._fix_data_quality_errors
        }
    
    def validate_knowledge_graph(self, kg_data: Dict[str, Any], auto_fix_errors: bool = True) -> Dict[str, Any]:
        """
        지식그래프 검증 수행
        
        Args:
            kg_data: 지식그래프 생성 결과
            
        Returns:
            검증 결과
        """
        try:
            if not kg_data.get('success'):
                return {
                    'success': False,
                    'error': 'Cannot validate failed knowledge graph generation'
                }
            
            # RDF 그래프 로드
            graph = Graph()
            turtle_rdf = kg_data['rdf_formats']['turtle']
            graph.parse(data=turtle_rdf, format='turtle')
            
            logger.info(f"Validating knowledge graph with {len(graph)} triples")
            
            validation_results = []
            overall_score = 0
            total_checks = len(self.validation_rules)
            
            # 각 검증 규칙 실행 및 오류 수정
            errors_fixed = []
            for rule in self.validation_rules:
                try:
                    result = rule(graph, kg_data)
                    
                    # 오류 자동 수정 시도
                    if not result['passed'] and auto_fix_errors:
                        rule_name = rule.__name__.replace('_check_', '')
                        if rule_name in self.error_fixing_rules:
                            try:
                                fix_result = self.error_fixing_rules[rule_name](graph, result)
                                if fix_result.get('success'):
                                    errors_fixed.append({
                                        'rule': rule_name,
                                        'fixes_applied': fix_result.get('fixes_applied', []),
                                        'message': fix_result.get('message')
                                    })
                                    # 수정 후 재검증
                                    result = rule(graph, kg_data)
                            except Exception as fix_error:
                                logger.error(f"Error fixing {rule_name}: {fix_error}")
                    
                    validation_results.append(result)
                    if result['passed']:
                        overall_score += result.get('score', 1)
                except Exception as e:
                    logger.error(f"Error in validation rule {rule.__name__}: {e}")
                    validation_results.append({
                        'rule': rule.__name__,
                        'passed': False,
                        'score': 0,
                        'message': f"Validation error: {str(e)}"
                    })
            
            # LLM을 사용한 추가 검증
            llm_validation = None
            if self.llm_engine:
                llm_validation = self._validate_with_llm(graph, kg_data)
            
            # 최종 점수 계산
            final_score = (overall_score / total_checks) * 100 if total_checks > 0 else 0
            
            # 수정된 그래프를 다시 직렬화
            updated_rdf_formats = None
            if errors_fixed:
                updated_rdf_formats = {
                    'turtle': graph.serialize(format='turtle'),
                    'xml': graph.serialize(format='xml'),
                    'json-ld': graph.serialize(format='json-ld'),
                    'n3': graph.serialize(format='n3')
                }
            
            return {
                'success': True,
                'overall_score': final_score,
                'total_checks': total_checks,
                'passed_checks': sum(1 for r in validation_results if r['passed']),
                'validation_results': validation_results,
                'llm_validation': llm_validation,
                'recommendations': self._generate_recommendations(validation_results),
                'errors_fixed': errors_fixed,
                'updated_rdf_formats': updated_rdf_formats,
                'auto_fix_enabled': auto_fix_errors,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating knowledge graph: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_basic_structure(self, graph: Graph, kg_data: Dict) -> Dict:
        """기본 구조 검사"""
        LLMIE = Namespace("http://llm-ie.org/ontology#")
        
        has_entities = len(list(graph.subjects(RDF.type, LLMIE.Entity))) > 0
        has_documents = len(list(graph.subjects(RDF.type, LLMIE.Document))) > 0
        has_labels = len(list(graph.objects(None, RDFS.label))) > 0
        
        passed = has_entities and has_documents and has_labels
        score = sum([has_entities, has_documents, has_labels]) / 3
        
        message = "Basic RDF structure is valid" if passed else "Missing basic RDF structure elements"
        
        return {
            'rule': 'basic_structure',
            'passed': passed,
            'score': score,
            'message': message,
            'details': {
                'has_entities': has_entities,
                'has_documents': has_documents,
                'has_labels': has_labels
            }
        }
    
    def _check_entity_consistency(self, graph: Graph, kg_data: Dict) -> Dict:
        """엔티티 일관성 검사"""
        LLMIE = Namespace("http://llm-ie.org/ontology#")
        
        entities = list(graph.subjects(RDF.type, LLMIE.Entity))
        inconsistencies = []
        
        for entity in entities:
            # 라벨 존재 확인
            labels = list(graph.objects(entity, RDFS.label))
            if not labels:
                inconsistencies.append(f"Entity {entity} has no label")
            
            # 위치 정보 확인
            start_pos = list(graph.objects(entity, LLMIE.startPosition))
            end_pos = list(graph.objects(entity, LLMIE.endPosition))
            if not start_pos or not end_pos:
                inconsistencies.append(f"Entity {entity} missing position information")
        
        passed = len(inconsistencies) == 0
        score = max(0, 1 - (len(inconsistencies) / max(len(entities), 1)))
        
        return {
            'rule': 'entity_consistency',
            'passed': passed,
            'score': score,
            'message': f"Found {len(inconsistencies)} entity inconsistencies" if inconsistencies else "All entities are consistent",
            'details': {'inconsistencies': inconsistencies}
        }
    
    def _check_relation_validity(self, graph: Graph, kg_data: Dict) -> Dict:
        """관계 유효성 검사"""
        statements = list(graph.subjects(RDF.type, RDF.Statement))
        invalid_relations = []
        
        for stmt in statements:
            subjects = list(graph.objects(stmt, RDF.subject))
            predicates = list(graph.objects(stmt, RDF.predicate))
            objects = list(graph.objects(stmt, RDF.object))
            
            if not (subjects and predicates and objects):
                invalid_relations.append(f"Incomplete relation: {stmt}")
        
        passed = len(invalid_relations) == 0
        score = max(0, 1 - (len(invalid_relations) / max(len(statements), 1)))
        
        return {
            'rule': 'relation_validity',
            'passed': passed,
            'score': score,
            'message': f"Found {len(invalid_relations)} invalid relations" if invalid_relations else "All relations are valid",
            'details': {'invalid_relations': invalid_relations}
        }
    
    def _check_namespace_usage(self, graph: Graph, kg_data: Dict) -> Dict:
        """네임스페이스 사용 검사"""
        expected_namespaces = [
            "http://llm-ie.org/ontology#",
            "http://llm-ie.org/entity#",
            "http://llm-ie.org/document#"
        ]
        
        used_namespaces = set()
        for s, p, o in graph:
            for uri in [s, p, o]:
                if isinstance(uri, URIRef):
                    namespace = str(uri).rsplit('#', 1)[0] + '#' if '#' in str(uri) else str(uri).rsplit('/', 1)[0] + '/'
                    used_namespaces.add(namespace)
        
        missing_namespaces = []
        for expected in expected_namespaces:
            if expected not in used_namespaces:
                missing_namespaces.append(expected)
        
        passed = len(missing_namespaces) == 0
        score = max(0, 1 - (len(missing_namespaces) / len(expected_namespaces)))
        
        return {
            'rule': 'namespace_usage',
            'passed': passed,
            'score': score,
            'message': f"Missing namespaces: {missing_namespaces}" if missing_namespaces else "All expected namespaces are used",
            'details': {
                'expected_namespaces': expected_namespaces,
                'used_namespaces': list(used_namespaces),
                'missing_namespaces': missing_namespaces
            }
        }
    
    def _check_data_quality(self, graph: Graph, kg_data: Dict) -> Dict:
        """데이터 품질 검사"""
        issues = []
        
        # 빈 라벨 검사
        for label in graph.objects(None, RDFS.label):
            if not str(label).strip():
                issues.append("Found empty label")
        
        # 중복 엔티티 검사 (같은 텍스트, 같은 위치)
        LLMIE = Namespace("http://llm-ie.org/ontology#")
        entities = list(graph.subjects(RDF.type, LLMIE.Entity))
        entity_signatures = {}
        
        for entity in entities:
            labels = list(graph.objects(entity, RDFS.label))
            start_pos = list(graph.objects(entity, LLMIE.startPosition))
            end_pos = list(graph.objects(entity, LLMIE.endPosition))
            
            if labels and start_pos and end_pos:
                signature = (str(labels[0]), str(start_pos[0]), str(end_pos[0]))
                if signature in entity_signatures:
                    issues.append(f"Duplicate entity: {labels[0]} at position {start_pos[0]}-{end_pos[0]}")
                entity_signatures[signature] = entity
        
        passed = len(issues) == 0
        score = max(0, 1 - (len(issues) / 10))  # 10개 이상의 이슈가 있으면 0점
        
        return {
            'rule': 'data_quality',
            'passed': passed,
            'score': score,
            'message': f"Found {len(issues)} data quality issues" if issues else "No data quality issues found",
            'details': {'issues': issues}
        }
    
    def _validate_with_llm(self, graph: Graph, kg_data: Dict) -> Dict:
        """LLM을 사용한 추가 검증"""
        if not self.llm_engine:
            return None
        
        try:
            # 그래프를 간단한 텍스트 표현으로 변환
            graph_summary = self._generate_graph_summary(graph)
            
            prompt = f"""
다음은 LLM-IE로 생성된 지식그래프입니다. 이 지식그래프의 품질과 일관성을 평가해주세요.

지식그래프 요약:
{graph_summary}

다음 항목들을 평가해주세요:
1. 엔티티 추출의 적절성 (1-10점)
2. 관계 추론의 논리성 (1-10점)  
3. 전체적인 일관성 (1-10점)
4. 실제 텍스트와의 정확성 (1-10점)

다음 JSON 형식으로 응답해주세요:
{{
    "entity_extraction_score": 8,
    "relation_logic_score": 7,
    "consistency_score": 9,
    "accuracy_score": 8,
    "overall_score": 8.0,
    "strengths": ["강점 1", "강점 2"],
    "weaknesses": ["약점 1", "약점 2"],
    "recommendations": ["추천사항 1", "추천사항 2"]
}}
"""
            
            messages = [{"role": "user", "content": prompt}]
            response_generator = self.llm_engine.chat_completion(messages, stream=False)
            response_text = ""
            
            for response in response_generator:
                if response.get('type') == 'content':
                    response_text += response.get('text', '')
            
            try:
                validation_result = json.loads(response_text.strip())
                return validation_result
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM validation response")
                return {
                    'error': 'Failed to parse LLM response',
                    'raw_response': response_text
                }
                
        except Exception as e:
            logger.error(f"Error in LLM validation: {e}")
            return {
                'error': str(e)
            }
    
    def _generate_graph_summary(self, graph: Graph) -> str:
        """그래프 요약 생성"""
        LLMIE = Namespace("http://llm-ie.org/ontology#")
        
        summary = []
        
        # 엔티티 요약
        entities = list(graph.subjects(RDF.type, LLMIE.Entity))
        summary.append(f"총 {len(entities)}개의 엔티티:")
        
        for entity in entities[:10]:  # 최대 10개만 표시
            labels = list(graph.objects(entity, RDFS.label))
            entity_types = [str(t).split('#')[-1] for t in graph.objects(entity, RDF.type) 
                           if t != LLMIE.Entity]
            if labels and entity_types:
                summary.append(f"  - {labels[0]} ({entity_types[0]})")
        
        if len(entities) > 10:
            summary.append(f"  ... 그 외 {len(entities) - 10}개")
        
        # 관계 요약
        statements = list(graph.subjects(RDF.type, RDF.Statement))
        summary.append(f"\n총 {len(statements)}개의 관계:")
        
        for stmt in statements[:5]:  # 최대 5개만 표시
            try:
                subject = list(graph.objects(stmt, RDF.subject))[0]
                predicate = list(graph.objects(stmt, RDF.predicate))[0]
                obj = list(graph.objects(stmt, RDF.object))[0]
                
                subj_label = list(graph.objects(subject, RDFS.label))
                obj_label = list(graph.objects(obj, RDFS.label))
                
                if subj_label and obj_label:
                    pred_name = str(predicate).split('#')[-1] if '#' in str(predicate) else str(predicate).split('/')[-1]
                    summary.append(f"  - {subj_label[0]} --{pred_name}--> {obj_label[0]}")
            except (IndexError, TypeError):
                continue
        
        if len(statements) > 5:
            summary.append(f"  ... 그 외 {len(statements) - 5}개")
        
        return '\n'.join(summary)
    
    def _generate_recommendations(self, validation_results: List[Dict]) -> List[str]:
        """검증 결과 기반 추천사항 생성"""
        recommendations = []
        
        for result in validation_results:
            if not result['passed']:
                rule = result['rule']
                if rule == 'basic_structure':
                    recommendations.append("기본 RDF 구조를 개선하세요: 모든 엔티티에 적절한 타입과 라벨을 추가하세요.")
                elif rule == 'entity_consistency':
                    recommendations.append("엔티티 일관성을 개선하세요: 모든 엔티티에 라벨과 위치 정보를 추가하세요.")
                elif rule == 'relation_validity':
                    recommendations.append("관계 유효성을 개선하세요: 모든 관계에 주어, 술어, 목적어가 완전히 정의되어야 합니다.")
                elif rule == 'namespace_usage':
                    recommendations.append("네임스페이스 사용을 개선하세요: 표준 네임스페이스를 일관성 있게 사용하세요.")
                elif rule == 'data_quality':
                    recommendations.append("데이터 품질을 개선하세요: 중복 엔티티를 제거하고 빈 값들을 정리하세요.")
        
        if not recommendations:
            recommendations.append("지식그래프가 모든 검증을 통과했습니다! 품질이 우수합니다.")
        
        # 자동 수정 기능에 대한 추천사항 추가
        recommendations.append("자동 오류 수정 기능을 사용하여 검출된 문제들을 자동으로 해결할 수 있습니다.")
        
        return recommendations
    
    def _fix_basic_structure_errors(self, graph: Graph, validation_result: Dict) -> Dict:
        """기본 구조 오류 수정"""
        LLMIE = Namespace("http://llm-ie.org/ontology#")
        fixes_applied = []
        
        try:
            details = validation_result.get('details', {})
            
            # 엔티티가 없는 경우 기본 엔티티 생성
            if not details.get('has_entities'):
                default_entity = self.ENTITY['default_entity']
                graph.add((default_entity, RDF.type, LLMIE.Entity))
                graph.add((default_entity, RDFS.label, Literal("Default Entity")))
                fixes_applied.append("Added default entity")
            
            # 문서가 없는 경우 기본 문서 생성
            if not details.get('has_documents'):
                default_doc = self.DOC['default_document']
                graph.add((default_doc, RDF.type, LLMIE.Document))
                graph.add((default_doc, RDFS.label, Literal("Default Document")))
                fixes_applied.append("Added default document")
            
            return {
                'success': True,
                'fixes_applied': fixes_applied,
                'message': f"Applied {len(fixes_applied)} basic structure fixes"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _fix_entity_consistency_errors(self, graph: Graph, validation_result: Dict) -> Dict:
        """엔티티 일관성 오류 수정"""
        LLMIE = Namespace("http://llm-ie.org/ontology#")
        fixes_applied = []
        
        try:
            entities = list(graph.subjects(RDF.type, LLMIE.Entity))
            
            for entity in entities:
                # 레이블이 없는 엔티티에 기본 레이블 추가
                labels = list(graph.objects(entity, RDFS.label))
                if not labels:
                    entity_id = str(entity).split('#')[-1] if '#' in str(entity) else str(entity).split('/')[-1]
                    graph.add((entity, RDFS.label, Literal(f"Entity_{entity_id}")))
                    fixes_applied.append(f"Added label to {entity_id}")
                
                # 위치 정보가 없는 엔티티에 기본 위치 추가
                start_pos = list(graph.objects(entity, LLMIE.startPosition))
                end_pos = list(graph.objects(entity, LLMIE.endPosition))
                
                if not start_pos:
                    graph.add((entity, LLMIE.startPosition, Literal(0, datatype=XSD.integer)))
                    fixes_applied.append(f"Added start position to {entity}")
                
                if not end_pos:
                    graph.add((entity, LLMIE.endPosition, Literal(0, datatype=XSD.integer)))
                    fixes_applied.append(f"Added end position to {entity}")
            
            return {
                'success': True,
                'fixes_applied': fixes_applied,
                'message': f"Applied {len(fixes_applied)} entity consistency fixes"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _fix_relation_validity_errors(self, graph: Graph, validation_result: Dict) -> Dict:
        """관계 유효성 오류 수정"""
        fixes_applied = []
        
        try:
            statements = list(graph.subjects(RDF.type, RDF.Statement))
            invalid_statements = []
            
            for stmt in statements:
                subjects = list(graph.objects(stmt, RDF.subject))
                predicates = list(graph.objects(stmt, RDF.predicate))
                objects = list(graph.objects(stmt, RDF.object))
                
                if not (subjects and predicates and objects):
                    invalid_statements.append(stmt)
            
            # 유효하지 않은 관계 제거
            for stmt in invalid_statements:
                # 관련된 모든 트리플 제거
                for s, p, o in list(graph.triples((stmt, None, None))):
                    graph.remove((s, p, o))
                fixes_applied.append(f"Removed invalid relation statement: {stmt}")
            
            return {
                'success': True,
                'fixes_applied': fixes_applied,
                'message': f"Applied {len(fixes_applied)} relation validity fixes"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _fix_namespace_usage_errors(self, graph: Graph, validation_result: Dict) -> Dict:
        """네임스페이스 사용 오류 수정"""
        fixes_applied = []
        
        try:
            # 기본 네임스페이스 바인딩 추가
            LLMIE = Namespace("http://llm-ie.org/ontology#")
            ENTITY = Namespace("http://llm-ie.org/entity#")
            DOC = Namespace("http://llm-ie.org/document#")
            
            graph.bind("llmie", LLMIE)
            graph.bind("entity", ENTITY)
            graph.bind("doc", DOC)
            graph.bind("rdf", RDF)
            graph.bind("rdfs", RDFS)
            graph.bind("foaf", FOAF)
            
            fixes_applied.append("Added standard namespace bindings")
            
            return {
                'success': True,
                'fixes_applied': fixes_applied,
                'message': f"Applied {len(fixes_applied)} namespace usage fixes"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _fix_data_quality_errors(self, graph: Graph, validation_result: Dict) -> Dict:
        """데이터 품질 오류 수정"""
        LLMIE = Namespace("http://llm-ie.org/ontology#")
        fixes_applied = []
        
        try:
            # 빈 레이블 제거
            empty_labels = []
            for s, p, o in graph.triples((None, RDFS.label, None)):
                if not str(o).strip():
                    empty_labels.append((s, p, o))
            
            for triple in empty_labels:
                graph.remove(triple)
                fixes_applied.append(f"Removed empty label from {triple[0]}")
            
            # 중복 엔티티 제거
            entities = list(graph.subjects(RDF.type, LLMIE.Entity))
            entity_signatures = {}
            duplicates_to_remove = []
            
            for entity in entities:
                labels = list(graph.objects(entity, RDFS.label))
                start_pos = list(graph.objects(entity, LLMIE.startPosition))
                end_pos = list(graph.objects(entity, LLMIE.endPosition))
                
                if labels and start_pos and end_pos:
                    signature = (str(labels[0]), str(start_pos[0]), str(end_pos[0]))
                    if signature in entity_signatures:
                        # 중복된 엔티티 마크
                        duplicates_to_remove.append(entity)
                    else:
                        entity_signatures[signature] = entity
            
            # 중복 엔티티 제거
            for duplicate in duplicates_to_remove:
                for s, p, o in list(graph.triples((duplicate, None, None))):
                    graph.remove((s, p, o))
                for s, p, o in list(graph.triples((None, None, duplicate))):
                    graph.remove((s, p, o))
                fixes_applied.append(f"Removed duplicate entity: {duplicate}")
            
            return {
                'success': True,
                'fixes_applied': fixes_applied,
                'message': f"Applied {len(fixes_applied)} data quality fixes"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }