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

    // 파일 업로드 처리
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    if (uploadZone) {
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
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
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

    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        if (!file.name.endsWith('.llmie')) {
            alert('LLM-IE 추출 결과 파일 (.llmie)만 지원됩니다.');
            return;
        }

        // 파일 정보 표시
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'block';

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

        // 네트워크 이벤트
        networkInstance.on('selectNode', (params) => {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                console.log('Selected node:', node);
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

    // 초기 상태 복원
    restoreState();

    console.log('Knowledge Graph module initialization complete');
});