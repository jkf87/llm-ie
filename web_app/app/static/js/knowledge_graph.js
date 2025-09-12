/**
 * Knowledge Graph 기능을 위한 JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('Knowledge Graph module loaded');

    // DOM 요소들
    const fileInput = document.getElementById('kg-file-input');
    const uploadZone = document.getElementById('kg-upload-zone');
    const fileInfo = document.getElementById('kg-file-info');
    const fileName = document.getElementById('kg-file-name');
    const fileSize = document.getElementById('kg-file-size');
    
    const generateBtn = document.getElementById('generate-kg-btn');
    const validateBtn = document.getElementById('validate-kg-btn');
    const clearBtn = document.getElementById('clear-kg-btn');
    
    const progressSection = document.getElementById('kg-progress');
    const progressFill = document.getElementById('kg-progress-fill');
    const progressText = document.getElementById('kg-progress-text');
    
    const resultsSection = document.getElementById('kg-results');
    const validationSection = document.getElementById('kg-validation-results');
    
    // 설정 요소들
    const enableLlmInference = document.getElementById('kg-enable-llm-inference');
    const llmApiSelect = document.getElementById('kg-llm-api-select');
    const geminiApiKey = document.getElementById('kg-gemini-api-key');
    
    // 전역 변수들
    let uploadedFileData = null;
    let currentKnowledgeGraph = null;
    let networkInstance = null;
    let networkNodes = null;
    let networkEdges = null;
    
    // 수동 관계 생성 관련 변수들
    let manualRelationMode = false;
    let selectedNodes = [];
    let currentRelationId = 1;

    // 파일 업로드 처리
    console.log('Setting up file upload handlers...');
    console.log('fileInput element:', fileInput);
    
    if (fileInput) {
        console.log('Adding change event listener to fileInput');
        fileInput.addEventListener('change', handleFileSelect);
    } else {
        console.error('fileInput element not found!');
    }
    
    if (uploadZone) {
        console.log('Setting up uploadZone event listeners');
        
        // 클릭으로 파일 선택 트리거
        uploadZone.addEventListener('click', (e) => {
            // 버튼 클릭이 아닌 영역 클릭시에만 파일 선택 트리거
            if (!e.target.matches('button') && fileInput) {
                console.log('uploadZone clicked, triggering file input');
                fileInput.click();
            }
        });
        
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
        });
        
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('drag-over');
        });
        
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            console.log('Files dropped:', e.dataTransfer.files);
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
    } else {
        console.error('uploadZone element not found!');
    }

    // 파일 선택 버튼 이벤트
    const fileSelectBtn = document.getElementById('kg-file-select-btn');
    if (fileSelectBtn && fileInput) {
        console.log('Setting up file select button');
        fileSelectBtn.addEventListener('click', () => {
            console.log('File select button clicked');
            fileInput.click();
        });
    } else {
        console.error('File select button or file input not found:', { fileSelectBtn, fileInput });
    }

    // 버튼 이벤트
    if (generateBtn) {
        generateBtn.addEventListener('click', generateKnowledgeGraph);
    }
    
    if (validateBtn) {
        validateBtn.addEventListener('click', validateKnowledgeGraph);
    }
    
    if (clearBtn) {
        clearBtn.addEventListener('click', clearResults);
    }

    // RDF 포맷 탭 처리
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('tab-btn')) {
            handleRdfFormatChange(e.target);
        }
    });

    // 복사 및 다운로드 버튼
    const copyRdfBtn = document.getElementById('copy-rdf-btn');
    const downloadRdfBtn = document.getElementById('download-rdf-btn');
    
    if (copyRdfBtn) {
        copyRdfBtn.addEventListener('click', copyRdfToClipboard);
    }
    
    if (downloadRdfBtn) {
        downloadRdfBtn.addEventListener('click', downloadRdf);
    }

    // 네트워크 시각화 컨트롤
    const fitNetworkBtn = document.getElementById('fit-network');
    const togglePhysicsBtn = document.getElementById('toggle-physics');
    
    if (fitNetworkBtn) {
        fitNetworkBtn.addEventListener('click', () => {
            if (networkInstance) {
                networkInstance.fit();
            }
        });
    }
    
    if (togglePhysicsBtn) {
        togglePhysicsBtn.addEventListener('click', () => {
            if (networkInstance) {
                const physicsEnabled = networkInstance.physics.physicsEnabled;
                networkInstance.setOptions({ physics: !physicsEnabled });
                togglePhysicsBtn.textContent = physicsEnabled ? '물리 효과 켜기' : '물리 효과 끄기';
            }
        });
    }
    
    // 수동 관계 생성 버튼들
    const manualRelationModeBtn = document.getElementById('manual-relation-mode-btn');
    const manualRelationControls = document.getElementById('manualRelationControls');
    const cancelManualRelationBtn = document.getElementById('cancelManualRelation');
    const clearSelectionBtn = document.getElementById('clearSelection');
    const createRelationBtn = document.getElementById('createRelationBtn');
    const relationConfidenceSlider = document.getElementById('relationConfidence');
    const confidenceValueSpan = document.getElementById('confidenceValue');
    
    if (manualRelationModeBtn) {
        manualRelationModeBtn.addEventListener('click', toggleManualRelationMode);
    }
    
    if (cancelManualRelationBtn) {
        cancelManualRelationBtn.addEventListener('click', exitManualRelationMode);
    }
    
    if (clearSelectionBtn) {
        clearSelectionBtn.addEventListener('click', clearNodeSelection);
    }
    
    if (createRelationBtn) {
        createRelationBtn.addEventListener('click', createManualRelation);
    }
    
    // 모달 폼에서 Enter 키 처리
    const manualRelationForm = document.getElementById('manualRelationForm');
    if (manualRelationForm) {
        manualRelationForm.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                createManualRelation();
            }
        });
    }
    
    if (relationConfidenceSlider && confidenceValueSpan) {
        relationConfidenceSlider.addEventListener('input', (e) => {
            confidenceValueSpan.textContent = e.target.value;
        });
    }
    
    // 노드 선택 드롭다운 이벤트 리스너
    const sourceNodeSelect = document.getElementById('sourceNodeSelect');
    const targetNodeSelect = document.getElementById('targetNodeSelect');
    
    if (sourceNodeSelect) {
        sourceNodeSelect.addEventListener('change', updateRelationPreview);
    }
    
    if (targetNodeSelect) {
        targetNodeSelect.addEventListener('change', updateRelationPreview);
    }
    
    // 관계 추출 컨텍스트 필드들
    const relationContextTextarea = document.getElementById('kg-relation-context');
    const relationGuidelinesTextarea = document.getElementById('kg-relation-guidelines');
    const relationTemplate = document.getElementById('kg-relation-template');
    const relationConfidenceRange = document.getElementById('kg-relation-confidence');
    const confidenceValue = document.getElementById('confidence-value');
    
    // 신뢰도 슬라이더 업데이트
    if (relationConfidenceRange && confidenceValue) {
        relationConfidenceRange.addEventListener('input', (e) => {
            confidenceValue.textContent = e.target.value;
        });
    }
    
    // 관계 템플릿 변경 시 컨텍스트 예시 업데이트
    if (relationTemplate && relationContextTextarea) {
        relationTemplate.addEventListener('change', (e) => {
            updateContextPlaceholder(e.target.value);
        });
    }

    function handleFileSelect(event) {
        console.log('handleFileSelect called', event);
        const file = event.target.files[0];
        console.log('Selected file:', file);
        if (file) {
            handleFile(file);
        } else {
            console.log('No file selected');
        }
    }

    function handleFile(file) {
        console.log('handleFile called with:', file);
        
        if (!file.name.endsWith('.llmie')) {
            alert('LLM-IE 추출 결과 파일 (.llmie)만 지원됩니다.');
            return;
        }

        console.log('File validation passed, updating UI...');
        
        // DOM 요소 존재 확인
        if (!fileName || !fileSize || !fileInfo) {
            console.error('Required DOM elements not found:', { fileName, fileSize, fileInfo });
            alert('페이지 요소를 찾을 수 없습니다. 페이지를 새로고침해주세요.');
            return;
        }

        // 파일 정보 표시
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'block';
        
        console.log('File info updated:', file.name, formatFileSize(file.size));

        // 파일 읽기
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                uploadedFileData = JSON.parse(e.target.result);
                generateBtn.disabled = false;
                console.log('File loaded successfully:', uploadedFileData);
            } catch (error) {
                alert('파일을 읽는 중 오류가 발생했습니다: ' + error.message);
                uploadedFileData = null;
                generateBtn.disabled = true;
            }
        };
        
        reader.onerror = () => {
            alert('파일을 읽는 중 오류가 발생했습니다.');
            uploadedFileData = null;
            generateBtn.disabled = true;
        };
        
        reader.readAsText(file);
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async function generateKnowledgeGraph() {
        if (!uploadedFileData) {
            alert('먼저 LLM-IE 추출 결과 파일을 업로드해주세요.');
            return;
        }

        // 진행상황 표시
        progressSection.style.display = 'block';
        resultsSection.style.display = 'none';
        validationSection.style.display = 'none';
        
        updateProgress(10, '지식그래프 생성 요청 준비 중...');

        try {
            // 요청 데이터 준비
            const requestData = {
                extraction_data: uploadedFileData,
                settings: {
                    enable_llm_inference: enableLlmInference.checked,
                    llm_api_type: llmApiSelect.value
                }
            };

            // 관계 추출 컨텍스트 설정 추가
            const relationDomain = document.getElementById('kg-domain-select');
            const relationTemplate = document.getElementById('kg-template-select'); 
            const relationContext = document.getElementById('kg-relation-context');
            const relationGoal = document.getElementById('kg-relation-goal');
            const relationConstraints = document.getElementById('kg-relation-guidelines');
            
            if (relationDomain && relationTemplate && relationContext && relationGoal && relationConstraints) {
                requestData.settings.relation_context = {
                    domain: relationDomain.value,
                    template: relationTemplate.value,
                    context: relationContext.value.trim(),
                    goal: relationGoal.value.trim(),
                    constraints: relationConstraints.value.trim()
                };
            }

            // Gemini API 키가 필요한 경우
            if (llmApiSelect.value === 'gemini_direct' && enableLlmInference.checked) {
                const apiKey = geminiApiKey.value.trim();
                if (!apiKey) {
                    alert('Gemini API 키를 입력해주세요.');
                    progressSection.style.display = 'none';
                    return;
                }
                requestData.settings.gemini_api_key = apiKey;
            }

            updateProgress(30, 'LLM-IE 서버에 지식그래프 생성 요청 중...');

            // API 요청
            const response = await fetch('/api/knowledge-graph/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            updateProgress(70, '응답 처리 중...');

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            const result = await response.json();

            updateProgress(90, '결과 렌더링 중...');

            if (result.success) {
                currentKnowledgeGraph = result;
                displayKnowledgeGraphResults(result);
                validateBtn.disabled = false;
                updateProgress(100, '지식그래프 생성 완료!');
                
                setTimeout(() => {
                    progressSection.style.display = 'none';
                }, 1000);
            } else {
                throw new Error(result.error || '지식그래프 생성에 실패했습니다.');
            }

        } catch (error) {
            console.error('Knowledge graph generation error:', error);
            alert('지식그래프 생성 중 오류가 발생했습니다: ' + error.message);
            progressSection.style.display = 'none';
        }
    }

    function updateProgress(percent, message) {
        progressFill.style.width = percent + '%';
        progressText.textContent = message;
    }

    function displayKnowledgeGraphResults(result) {
        // 통계 정보 업데이트
        document.getElementById('total-triples').textContent = result.total_triples || 0;
        document.getElementById('total-entities').textContent = result.statistics.total_entities || 0;
        document.getElementById('total-relations').textContent = result.statistics.total_relations || 0;
        document.getElementById('total-documents').textContent = result.statistics.total_documents || 0;

        // 시각화 생성
        if (result.visualization_data) {
            createNetworkVisualization(result.visualization_data);
        }

        // RDF 출력 표시 (기본값: Turtle)
        displayRdfOutput(result.rdf_formats, 'turtle');

        // 결과 섹션 표시
        resultsSection.style.display = 'block';
    }

    function createNetworkVisualization(vizData) {
        const container = document.getElementById('kg-visualization');
        
        if (!vizData.nodes || !vizData.edges) {
            container.innerHTML = '<p class="text-center text-muted">시각화 데이터가 없습니다.</p>';
            return;
        }

        // vis.js가 로드되어 있는지 확인
        if (typeof vis === 'undefined') {
            container.innerHTML = '<p class="text-center text-warning">시각화 라이브러리를 로드할 수 없습니다.</p>';
            return;
        }

        // 노드 데이터 준비
        const nodes = new vis.DataSet(vizData.nodes.map(node => ({
            id: node.id,
            label: node.label,
            group: node.group,
            title: `${node.label} (${node.type})`,
            color: getNodeColor(node.type)
        })));

        // 엣지 데이터 준비
        const edges = new vis.DataSet(vizData.edges.map(edge => ({
            from: edge.from,
            to: edge.to,
            label: edge.label,
            title: `${edge.label} (신뢰도: ${(edge.confidence * 100).toFixed(1)}%)`,
            width: Math.max(1, edge.confidence * 5)
        })));

        const data = { nodes, edges };

        const options = {
            physics: {
                enabled: true,
                stabilization: { enabled: true, iterations: 100 }
            },
            nodes: {
                shape: 'dot',
                size: 15,
                font: { size: 12, color: '#333' },
                borderWidth: 2
            },
            edges: {
                arrows: { to: { enabled: true, scaleFactor: 1, type: 'arrow' } },
                color: { inherit: 'from' },
                font: { size: 10, align: 'middle' },
                smooth: { enabled: true, type: 'continuous' }
            },
            interaction: {
                hover: true,
                tooltipDelay: 300,
                hideEdgesOnDrag: true
            },
            layout: {
                improvedLayout: true
            }
        };

        networkInstance = new vis.Network(container, data, options);

        // 노드와 엣지 데이터를 전역 변수에 저장 (수동 관계 생성용)
        networkNodes = nodes;
        networkEdges = edges;
        
        // 네트워크 이벤트
        networkInstance.on('selectNode', (params) => {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                console.log('Selected node:', node);
                
                // 수동 관계 생성 모드인 경우 특별 처리
                if (manualRelationMode) {
                    handleNodeClick(params);
                }
            }
        });

        networkInstance.on('selectEdge', (params) => {
            if (params.edges.length > 0) {
                const edgeId = params.edges[0];
                const edge = edges.get(edgeId);
                console.log('Selected edge:', edge);
            }
        });
    }

    function getNodeColor(nodeType) {
        const colorMap = {
            'MODEL': '#FF6B6B',
            'DOCUMENT': '#4ECDC4',
            'TECHNOLOGY': '#45B7D1',
            'TECHNOLOGY_CONCEPT': '#96CEB4',
            'TECHNOLOGY_PARAMETER': '#FECA57',
            'SURVEY': '#FF9FF3',
            'RESEARCH': '#54A0FF',
            'CONCEPT': '#5F27CD',
            'UNKNOWN': '#C4C4C4'
        };
        
        return colorMap[nodeType] || colorMap['UNKNOWN'];
    }

    function displayRdfOutput(rdfFormats, activeFormat) {
        const rdfOutput = document.getElementById('rdf-output');
        
        if (rdfFormats && rdfFormats[activeFormat]) {
            rdfOutput.textContent = rdfFormats[activeFormat];
        } else {
            rdfOutput.textContent = 'RDF 데이터가 없습니다.';
        }

        // 활성 탭 업데이트
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.format === activeFormat);
        });
    }

    function handleRdfFormatChange(tabBtn) {
        const format = tabBtn.dataset.format;
        if (currentKnowledgeGraph && currentKnowledgeGraph.rdf_formats) {
            displayRdfOutput(currentKnowledgeGraph.rdf_formats, format);
        }
    }

    async function copyRdfToClipboard() {
        const rdfOutput = document.getElementById('rdf-output');
        const text = rdfOutput.textContent;
        
        try {
            await navigator.clipboard.writeText(text);
            
            // 버튼 피드백
            const originalText = copyRdfBtn.textContent;
            copyRdfBtn.textContent = '✅ 복사됨!';
            copyRdfBtn.disabled = true;
            
            setTimeout(() => {
                copyRdfBtn.textContent = originalText;
                copyRdfBtn.disabled = false;
            }, 2000);
        } catch (error) {
            alert('클립보드 복사에 실패했습니다.');
        }
    }

    function downloadRdf() {
        const rdfOutput = document.getElementById('rdf-output');
        const text = rdfOutput.textContent;
        const activeFormat = document.querySelector('.tab-btn.active')?.dataset.format || 'turtle';
        
        // 파일 확장자 매핑
        const extensionMap = {
            'turtle': 'ttl',
            'xml': 'rdf',
            'json-ld': 'jsonld',
            'n3': 'n3'
        };
        
        const extension = extensionMap[activeFormat] || 'rdf';
        const filename = `knowledge_graph_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.${extension}`;
        
        const blob = new Blob([text], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    async function validateKnowledgeGraph() {
        if (!currentKnowledgeGraph) {
            alert('먼저 지식그래프를 생성해주세요.');
            return;
        }

        progressSection.style.display = 'block';
        updateProgress(10, '지식그래프 검증 시작...');

        try {
            const requestData = {
                kg_data: currentKnowledgeGraph,
                settings: {
                    enable_llm_validation: enableLlmInference.checked,
                    llm_api_type: llmApiSelect.value
                }
            };

            // Gemini API 키가 필요한 경우
            if (llmApiSelect.value === 'gemini_direct' && enableLlmInference.checked) {
                const apiKey = geminiApiKey.value.trim();
                if (!apiKey) {
                    alert('LLM 검증을 위해 Gemini API 키를 입력해주세요.');
                    progressSection.style.display = 'none';
                    return;
                }
                requestData.settings.gemini_api_key = apiKey;
            }

            updateProgress(50, '지식그래프 검증 실행 중...');

            const response = await fetch('/api/knowledge-graph/validate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            updateProgress(80, '검증 결과 처리 중...');

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            const result = await response.json();

            if (result.success) {
                displayValidationResults(result);
                updateProgress(100, '검증 완료!');
                
                setTimeout(() => {
                    progressSection.style.display = 'none';
                }, 1000);
            } else {
                throw new Error(result.error || '지식그래프 검증에 실패했습니다.');
            }

        } catch (error) {
            console.error('Knowledge graph validation error:', error);
            alert('지식그래프 검증 중 오류가 발생했습니다: ' + error.message);
            progressSection.style.display = 'none';
        }
    }

    function displayValidationResults(result) {
        // 전체 점수 표시
        const scoreValue = document.getElementById('validation-score');
        const scoreSummary = document.getElementById('validation-summary');
        
        scoreValue.textContent = Math.round(result.overall_score);
        scoreSummary.textContent = `${result.passed_checks}/${result.total_checks} 검사 통과`;

        // 점수에 따른 색상 변경
        const scoreCircle = document.querySelector('.score-circle');
        if (result.overall_score >= 80) {
            scoreCircle.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
        } else if (result.overall_score >= 60) {
            scoreCircle.style.background = 'linear-gradient(135deg, #ffc107, #fd7e14)';
        } else {
            scoreCircle.style.background = 'linear-gradient(135deg, #dc3545, #e83e8c)';
        }

        // 세부 검증 결과 표시
        displayValidationChecks(result.validation_results);

        // LLM 검증 결과 표시
        if (result.llm_validation) {
            displayLlmValidation(result.llm_validation);
        }

        // 검증 섹션 표시
        validationSection.style.display = 'block';
    }

    function displayValidationChecks(validationResults) {
        const checksContainer = document.getElementById('validation-checks');
        checksContainer.innerHTML = '';

        validationResults.forEach(check => {
            const checkDiv = document.createElement('div');
            checkDiv.className = `validation-check ${check.passed ? 'passed' : 'failed'}`;
            
            const icon = check.passed ? '✅' : '❌';
            const ruleName = check.rule.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            
            checkDiv.innerHTML = `
                <span class="check-icon ${check.passed ? 'passed' : 'failed'}">${icon}</span>
                <div class="check-content">
                    <strong>${ruleName}</strong>
                    <p>${check.message}</p>
                    <small>점수: ${Math.round(check.score * 100)}/100</small>
                </div>
            `;
            
            checksContainer.appendChild(checkDiv);
        });
    }

    function displayLlmValidation(llmValidation) {
        const llmPanel = document.getElementById('llm-validation-panel');
        
        if (llmValidation.error) {
            llmPanel.innerHTML = `<p class="text-warning">LLM 검증 중 오류 발생: ${llmValidation.error}</p>`;
            llmPanel.style.display = 'block';
            return;
        }

        // LLM 점수 표시
        const scoresContainer = document.getElementById('llm-scores');
        scoresContainer.innerHTML = '';

        const scores = [
            { label: '엔티티 추출', value: llmValidation.entity_extraction_score },
            { label: '관계 추론', value: llmValidation.relation_logic_score },
            { label: '일관성', value: llmValidation.consistency_score },
            { label: '정확성', value: llmValidation.accuracy_score }
        ];

        scores.forEach(score => {
            const scoreDiv = document.createElement('div');
            scoreDiv.className = 'llm-score-item';
            scoreDiv.innerHTML = `
                <div class="stat-label">${score.label}</div>
                <div class="stat-value">${score.value}/10</div>
            `;
            scoresContainer.appendChild(scoreDiv);
        });

        // 피드백 표시
        displayFeedbackList('validation-strengths', llmValidation.strengths);
        displayFeedbackList('validation-weaknesses', llmValidation.weaknesses);
        displayFeedbackList('validation-recommendations', llmValidation.recommendations);

        llmPanel.style.display = 'block';
    }

    function displayFeedbackList(containerId, items) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';

        if (!items || items.length === 0) {
            container.innerHTML = '<li>없음</li>';
            return;
        }

        items.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            container.appendChild(li);
        });
    }

    function clearResults() {
        // 파일 업로드 초기화
        fileInput.value = '';
        fileInfo.style.display = 'none';
        uploadedFileData = null;
        currentKnowledgeGraph = null;

        // 버튼 상태 초기화
        generateBtn.disabled = true;
        validateBtn.disabled = true;

        // 결과 섹션 숨기기
        progressSection.style.display = 'none';
        resultsSection.style.display = 'none';
        validationSection.style.display = 'none';

        // 네트워크 인스턴스 정리
        if (networkInstance) {
            networkInstance.destroy();
            networkInstance = null;
        }

        console.log('Results cleared');
    }

    // 상태 저장/복원 (localStorage 사용)
    function saveState() {
        const state = {
            enable_llm_inference: enableLlmInference.checked,
            llm_api_select: llmApiSelect.value,
            gemini_api_key: geminiApiKey.value
        };
        localStorage.setItem('kg_settings', JSON.stringify(state));
    }

    function restoreState() {
        const savedState = localStorage.getItem('kg_settings');
        if (savedState) {
            try {
                const state = JSON.parse(savedState);
                if (typeof state.enable_llm_inference === 'boolean') {
                    enableLlmInference.checked = state.enable_llm_inference;
                }
                if (state.llm_api_select) {
                    llmApiSelect.value = state.llm_api_select;
                }
                if (state.gemini_api_key) {
                    geminiApiKey.value = state.gemini_api_key;
                }
            } catch (error) {
                console.warn('Failed to restore state:', error);
            }
        }
    }

    // 설정 변경 시 저장
    [enableLlmInference, llmApiSelect, geminiApiKey].forEach(element => {
        if (element) {
            element.addEventListener('change', saveState);
        }
    });

    // 수동 관계 생성 함수들
    function toggleManualRelationMode() {
        // 직접 모달 표시
        showManualRelationModal();
    }
    
    function enterManualRelationMode() {
        console.log('Entering manual relation mode');
        manualRelationMode = true;
        clearNodeSelection();
        
        // UI 업데이트
        if (manualRelationModeBtn) {
            manualRelationModeBtn.textContent = '❌ 관계 생성 종료';
            manualRelationModeBtn.classList.remove('btn-outline-success');
            manualRelationModeBtn.classList.add('btn-outline-danger');
        }
        
        if (manualRelationControls) {
            manualRelationControls.style.display = 'block';
        }
        
        // 네트워크 시각화 컨테이너에 모드 표시
        const vizContainer = document.getElementById('kg-visualization');
        if (vizContainer) {
            vizContainer.classList.add('manual-relation-mode');
        }
        
        updateNodeSelectionDisplay();
    }
    
    function exitManualRelationMode() {
        console.log('Exiting manual relation mode');
        manualRelationMode = false;
        clearNodeSelection();
        
        // UI 업데이트
        if (manualRelationModeBtn) {
            manualRelationModeBtn.textContent = '🔗 수동 관계 생성';
            manualRelationModeBtn.classList.remove('btn-outline-danger');
            manualRelationModeBtn.classList.add('btn-outline-success');
        }
        
        if (manualRelationControls) {
            manualRelationControls.style.display = 'none';
        }
        
        // 네트워크 시각화 컨테이너에서 모드 표시 제거
        const vizContainer = document.getElementById('kg-visualization');
        if (vizContainer) {
            vizContainer.classList.remove('manual-relation-mode');
        }
    }
    
    function clearNodeSelection() {
        selectedNodes = [];
        updateNodeSelectionDisplay();
    }
    
    function updateNodeSelectionDisplay() {
        const selectedNodesList = document.getElementById('selectedNodesList');
        if (selectedNodesList) {
            if (selectedNodes.length === 0) {
                selectedNodesList.textContent = '없음';
            } else {
                const nodeLabels = selectedNodes.map(nodeId => {
                    const node = networkNodes ? networkNodes.get(nodeId) : null;
                    return node ? node.label : nodeId;
                });
                selectedNodesList.innerHTML = nodeLabels.map(label => 
                    `<span class="selected-node">${label}</span>`
                ).join(' ');
            }
        }
    }
    
    function handleNodeClick(params) {
        // 노드 클릭 시 특별한 동작이 필요한 경우 여기에 추가
        return;
    }
    
    function showManualRelationModal() {
        // 노드 선택 드롭다운 채우기
        populateNodeSelects();
        
        // 기존 선택된 노드가 있다면 드롭다운에 설정
        if (selectedNodes.length >= 1) {
            const sourceSelect = document.getElementById('sourceNodeSelect');
            if (sourceSelect) sourceSelect.value = selectedNodes[0];
        }
        if (selectedNodes.length >= 2) {
            const targetSelect = document.getElementById('targetNodeSelect');
            if (targetSelect) targetSelect.value = selectedNodes[1];
            updateRelationPreview();
        }
        
        // 폼 초기화
        const relationForm = document.getElementById('manualRelationForm');
        if (relationForm) {
            relationForm.reset();
            // 신뢰도를 기본값으로 설정
            const confidenceSlider = document.getElementById('relationConfidence');
            const confidenceValue = document.getElementById('confidenceValue');
            if (confidenceSlider && confidenceValue) {
                confidenceSlider.value = '0.9';
                confidenceValue.textContent = '0.9';
            }
        }
        
        // 모달 표시
        const modal = new bootstrap.Modal(document.getElementById('manualRelationModal'));
        modal.show();
        
        // 모달이 완전히 표시된 후 첫 번째 드롭다운에 포커스
        document.getElementById('manualRelationModal').addEventListener('shown.bs.modal', function () {
            const sourceSelect = document.getElementById('sourceNodeSelect');
            if (sourceSelect && !sourceSelect.value) {
                sourceSelect.focus();
            } else {
                const relationLabel = document.getElementById('relationLabel');
                if (relationLabel) {
                    relationLabel.focus();
                }
            }
        }, { once: true });
    }
    
    function populateNodeSelects() {
        const sourceSelect = document.getElementById('sourceNodeSelect');
        const targetSelect = document.getElementById('targetNodeSelect');
        
        if (!sourceSelect || !targetSelect || !networkNodes) return;
        
        // 기존 옵션 제거 (첫 번째 placeholder 옵션 제외)
        sourceSelect.innerHTML = '<option value="">시작 노드를 선택하세요</option>';
        targetSelect.innerHTML = '<option value="">대상 노드를 선택하세요</option>';
        
        // 모든 노드를 드롭다운에 추가
        networkNodes.forEach((node, nodeId) => {
            const option1 = new Option(node.label, nodeId);
            const option2 = new Option(node.label, nodeId);
            sourceSelect.appendChild(option1);
            targetSelect.appendChild(option2);
        });
    }
    
    function updateRelationPreview() {
        const sourceSelect = document.getElementById('sourceNodeSelect');
        const targetSelect = document.getElementById('targetNodeSelect');
        const relationPreview = document.getElementById('relationPreview');
        
        if (!sourceSelect || !targetSelect || !relationPreview) return;
        
        const sourceNodeId = sourceSelect.value;
        const targetNodeId = targetSelect.value;
        
        if (sourceNodeId && targetNodeId && sourceNodeId !== targetNodeId) {
            const sourceNode = networkNodes ? networkNodes.get(sourceNodeId) : null;
            const targetNode = networkNodes ? networkNodes.get(targetNodeId) : null;
            
            if (sourceNode && targetNode) {
                // 미리보기 업데이트
                const sourceNodePreview = document.getElementById('sourceNodePreview');
                const targetNodePreview = document.getElementById('targetNodePreview');
                
                if (sourceNodePreview) sourceNodePreview.textContent = sourceNode.label;
                if (targetNodePreview) targetNodePreview.textContent = targetNode.label;
                
                relationPreview.style.display = 'block';
                
                // selectedNodes 업데이트 (기존 로직과 호환성 위해)
                selectedNodes = [sourceNodeId, targetNodeId];
            } else {
                relationPreview.style.display = 'none';
            }
        } else {
            relationPreview.style.display = 'none';
        }
    }
    
    async function createManualRelation() {
        if (selectedNodes.length !== 2) return;
        
        const relationLabel = document.getElementById('relationLabel').value.trim();
        const relationDescription = document.getElementById('relationDescription').value.trim();
        const relationConfidence = parseFloat(document.getElementById('relationConfidence').value);
        const relationDirection = document.getElementById('relationDirection').value;
        
        if (!relationLabel) {
            alert('관계 라벨을 입력해주세요.');
            return;
        }
        
        // 관계 생성 버튼 비활성화
        const createBtn = document.getElementById('createRelationBtn');
        if (createBtn) {
            createBtn.disabled = true;
            createBtn.textContent = '생성 중...';
        }
        
        try {
            // 새로운 관계 데이터
            const relationData = {
                id: `manual_rel_${currentRelationId++}`,
                from: selectedNodes[0],
                to: selectedNodes[1],
                label: relationLabel,
                description: relationDescription,
                confidence: relationConfidence,
                direction: relationDirection
            };
            
            // 시각화용 관계 객체
            const newRelation = {
                id: relationData.id,
                from: relationData.from,
                to: relationData.to,
                label: relationData.label,
                title: `${relationData.label}${relationData.description ? ': ' + relationData.description : ''} (신뢰도: ${(relationData.confidence * 100).toFixed(1)}%)`,
                width: Math.max(1, relationData.confidence * 5),
                color: { color: '#4CAF50', highlight: '#45a049' }, // 수동 생성 관계는 녹색
                arrows: relationData.direction === 'directed' ? { to: { enabled: true, scaleFactor: 1, type: 'arrow' } } : false,
                manual: true,
                description: relationData.description,
                confidence: relationData.confidence
            };
            
            // 무방향성 관계인 경우 양방향 화살표 추가
            if (relationData.direction === 'undirected') {
                newRelation.arrows = { 
                    to: { enabled: true, scaleFactor: 1, type: 'arrow' },
                    from: { enabled: true, scaleFactor: 1, type: 'arrow' }
                };
            }
            
            // 즉시 시각화에 관계 추가
            if (networkEdges) {
                networkEdges.add(newRelation);
                console.log('Manual relation created:', newRelation);
            }
            
            // 선택적으로 서버에 저장 (현재는 로컬에서만 처리)
            // 향후 필요시 서버 API 호출 가능:
            /*
            const response = await fetch('/api/knowledge-graph/manual-relation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ relation: relationData })
            });
            
            if (!response.ok) {
                throw new Error('서버에 관계를 저장하는데 실패했습니다.');
            }
            */
            
            // 성공 메시지
            console.log('Manual relation added successfully');
            
        } catch (error) {
            console.error('Error creating manual relation:', error);
            alert('관계 생성 중 오류가 발생했습니다: ' + error.message);
            
            // 실패한 경우 추가된 관계 제거
            if (networkEdges && relationData) {
                networkEdges.remove(relationData.id);
            }
            
        } finally {
            // 버튼 상태 복원
            if (createBtn) {
                createBtn.disabled = false;
                createBtn.textContent = '관계 생성';
            }
            
            // 모달 닫기
            const modal = bootstrap.Modal.getInstance(document.getElementById('manualRelationModal'));
            if (modal) {
                modal.hide();
            }
            
            // 선택 초기화
            clearNodeSelection();
        }
    }

    // 컨텍스트 플레이스홀더 업데이트 함수
    function updateContextPlaceholder() {
        const domainSelect = document.getElementById('kg-domain-select');
        const templateSelect = document.getElementById('kg-template-select');
        const contextTextarea = document.getElementById('kg-relation-context');
        const goalTextarea = document.getElementById('kg-relation-goal');
        const constraintsTextarea = document.getElementById('kg-relation-guidelines');
        
        if (!domainSelect || !templateSelect || !contextTextarea || !goalTextarea || !constraintsTextarea) {
            return;
        }
        
        const domain = domainSelect.value;
        const template = templateSelect.value;
        
        // 도메인별 컨텍스트 예시
        const domainExamples = {
            'academic': {
                context: '이 문서는 학술 논문이며, 연구 방법론, 실험 결과, 이론적 배경 등을 다룹니다.',
                goal: '연구 주제, 방법론, 저자, 기관 간의 학술적 관계를 식별하고 연구 네트워크를 구성합니다.',
                constraints: '인용 관계, 공동 연구 관계, 이론적 영향 관계에 중점을 두고, 추측성 관계는 제외합니다.'
            },
            'business': {
                context: '이 문서는 비즈니스/기업 관련 내용으로, 조직, 제품, 서비스, 시장 등을 다룹니다.',
                goal: '기업 간 파트너십, 경쟁 관계, 공급망, 투자 관계 등 비즈니스 생태계를 파악합니다.',
                constraints: '계약 관계, 소유 관계, 경쟁 관계, 협력 관계에 집중하고, 루머나 추측은 배제합니다.'
            },
            'technical': {
                context: '이 문서는 기술/시스템 관련 내용으로, 소프트웨어, 하드웨어, 프로토콜 등을 다룹니다.',
                goal: '시스템 컴포넌트 간 의존성, 통신 관계, 상속 관계 등 기술적 아키텍처를 구성합니다.',
                constraints: 'API 의존성, 데이터 흐름, 상속/구현 관계, 통신 프로토콜에 중점을 두고 명확한 기술적 관계만 추출합니다.'
            },
            'general': {
                context: '이 문서는 일반적인 정보를 담고 있으며, 다양한 주제를 포괄합니다.',
                goal: '엔티티 간의 명확한 관계를 식별하여 정보의 구조와 연결성을 파악합니다.',
                constraints: '명시적으로 언급된 관계에만 집중하고, 애매하거나 추론적인 관계는 제외합니다.'
            }
        };
        
        // 템플릿별 추가 지침
        const templateGuidance = {
            'detailed': {
                goal_suffix: ' 관계의 세부사항과 맥락을 포함하여 상세하게 추출합니다.',
                constraints_suffix: ' 관계의 강도, 방향성, 시간적 맥락을 고려하여 정확도를 높입니다.'
            },
            'concise': {
                goal_suffix: ' 핵심적인 관계만을 간결하게 식별합니다.',
                constraints_suffix: ' 명확하고 직접적인 관계에만 집중하여 노이즈를 최소화합니다.'
            },
            'exploratory': {
                goal_suffix: ' 다양한 관계 유형을 탐색하여 숨겨진 연결점을 발견합니다.',
                constraints_suffix: ' 직접 관계뿐만 아니라 간접적 연관성도 고려하되 신뢰도 기준을 유지합니다.'
            }
        };
        
        // 플레이스홀더 업데이트
        const domainData = domainExamples[domain] || domainExamples['general'];
        const templateData = templateGuidance[template] || templateGuidance['detailed'];
        
        contextTextarea.placeholder = domainData.context;
        goalTextarea.placeholder = domainData.goal + (templateData.goal_suffix || '');
        constraintsTextarea.placeholder = domainData.constraints + (templateData.constraints_suffix || '');
    }

    // 관계 컨텍스트 필드 이벤트 리스너 설정
    const relationDomainSelect = document.getElementById('kg-domain-select');
    const relationTemplateSelect = document.getElementById('kg-template-select');
    
    if (relationDomainSelect) {
        relationDomainSelect.addEventListener('change', updateContextPlaceholder);
    }
    
    if (relationTemplateSelect) {
        relationTemplateSelect.addEventListener('change', updateContextPlaceholder);
    }
    
    // 초기 플레이스홀더 설정
    updateContextPlaceholder();

    // 초기 상태 복원
    restoreState();

    console.log('Knowledge Graph module initialization complete');
});