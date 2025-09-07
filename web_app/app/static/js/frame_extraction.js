// static/js/frame_extraction.js
document.addEventListener('DOMContentLoaded', () => {
    // --- UI Element Selectors ---
    const inputTextElem = document.getElementById('fe-input-text');
    const promptTemplateElem = document.getElementById('fe-prompt-template');
    const extractionUnitElem = document.getElementById('fe-extraction-unit');
    const contextElem = document.getElementById('fe-context');
    const temperatureElem = document.getElementById('fe-temperature');
    const maxTokensElem = document.getElementById('fe-max-tokens');
    const fuzzyMatchElem = document.getElementById('fe-fuzzy-match');

    const startButton = document.getElementById('start-extraction-btn');
    const startBatchButton = document.getElementById('start-batch-extraction-btn');
    const clearButton = document.getElementById('clear-extraction-btn');
    const downloadButton = document.getElementById('download-frames-btn');
    const outputElem = document.getElementById('extraction-output');
    const displayInputElem = document.getElementById('display-input-text');
    
    // 배치 처리 UI 요소들
    const enableBatchCheckbox = document.getElementById('fe-enable-batch');
    const batchOptionsDiv = document.getElementById('fe-batch-options');
    const batchTypeLlmieRadio = document.getElementById('fe-batch-type-llmie');
    const batchTypeGeminiRadio = document.getElementById('fe-batch-type-gemini');
    const apiKeysTextarea = document.getElementById('fe-api-keys');
    const chunkSizeInput = document.getElementById('fe-chunk-size');
    const overlapSizeInput = document.getElementById('fe-overlap-size');
    const batchSizeInput = document.getElementById('fe-batch-size');
    const delayBetweenBatchesInput = document.getElementById('fe-delay-between-batches');

    const feLlmApiSelect = document.getElementById('fe-llm-api-select');
    const feLlmConfigTypeSelect = document.getElementById('fe-llm-config-type-select');

    // Conditional LLM API Option Elements
    const feOpenaiCompatibleApiKey = document.getElementById('fe-openai-compatible-api-key');
    const feLlmBaseUrl = document.getElementById('fe-llm-base-url');
    const feLlmModelOpenaiComp = document.getElementById('fe-llm-model-openai-comp');
    const feOllamaHost = document.getElementById('fe-ollama-host');
    const feOllamaModel = document.getElementById('fe-ollama-model');
    const feOllamaNumCtx = document.getElementById('fe-ollama-num-ctx');
    const feHfToken = document.getElementById('fe-hf-token');
    const feHfModelOrEndpoint = document.getElementById('fe-hf-model-or-endpoint');
    const feOpenaiApiKey = document.getElementById('fe-openai-api-key');
    const feOpenaiModel = document.getElementById('fe-openai-model');
    // const feOpenaiReasoningModel = document.getElementById('fe-openai-reasoning-model'); // REMOVED
    const feAzureOpenaiApiKey = document.getElementById('fe-azure-openai-api-key');
    const feAzureEndpoint = document.getElementById('fe-azure-endpoint');
    const feAzureApiVersion = document.getElementById('fe-azure-api-version');
    const feAzureDeploymentName = document.getElementById('fe-azure-deployment-name');
    // const feAzureReasoningModel = document.getElementById('fe-azure-reasoning-model'); // REMOVED
    const feLitellmModel = document.getElementById('fe-litellm-model');
    const feLitellmApiKey = document.getElementById('fe-litellm-api-key');
    const feLitellmBaseUrl = document.getElementById('fe-litellm-base-url');
    const feGeminiApiKey = document.getElementById('fe-gemini-api-key');
    const feGeminiModel = document.getElementById('fe-gemini-model');
    
    const allFeApiOptionElements = [
        feOpenaiCompatibleApiKey, feLlmBaseUrl, feLlmModelOpenaiComp,
        feOllamaHost, feOllamaModel, feOllamaNumCtx,
        feHfToken, feHfModelOrEndpoint,
        feOpenaiApiKey, feOpenaiModel, // feOpenaiReasoningModel reference removed
        feAzureOpenaiApiKey, feAzureEndpoint, feAzureApiVersion, feAzureDeploymentName, // feAzureReasoningModel reference removed
        feLitellmModel, feLitellmApiKey, feLitellmBaseUrl,
        feGeminiApiKey, feGeminiModel
    ].filter(el => el); 

    // Conditional LLM Config Type Option Elements for Frame Extraction
    const feOpenAIReasoningOptionsDiv = document.getElementById('fe-openai_reasoning-config-options');
    const feOpenAIReasoningEffortSelect = document.getElementById('fe-openai-reasoning-effort');
    const feQwen3OptionsDiv = document.getElementById('fe-qwen3-config-options');
    const feQwenThinkingModeCheckbox = document.getElementById('fe-qwen-thinking-mode');

    let currentExtractedFrames = null; 

    // --- Helper Functions ---
    function escapeHTML(str) {
        if (typeof str !== 'string') {
            try {
                str = JSON.stringify(str, null, 2); 
            } catch (e) {
                str = String(str); 
            }
        }
        const div = document.createElement('div');
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    // --- LLM API Configuration UI Logic ---
    function feUpdateConditionalOptions() { 
        if (!feLlmApiSelect) return;
        const selectedApi = feLlmApiSelect.value;
        document.querySelectorAll('#fe-llm-config-form .conditional-options').forEach(div => {
            div.style.display = 'none';
        });
        if (selectedApi) {
            const targetDivId = `fe-${selectedApi}-options`;
            const optionsDiv = document.getElementById(targetDivId);
            if (optionsDiv) {
                optionsDiv.style.display = 'block';
            } else {
                console.warn(`FrameExtraction: updateConditionalOptions: Could not find div for ID: ${targetDivId}`);
            }
        }
        if (selectedApi) feHideApiSelectionWarning();
    }
    
    // --- LLM Config Type Configuration UI Logic ---
    function feUpdateConditionalLLMConfigOptions() { 
        if (!feLlmConfigTypeSelect) return;
        const selectedConfigType = feLlmConfigTypeSelect.value;

        if (feOpenAIReasoningOptionsDiv) feOpenAIReasoningOptionsDiv.style.display = 'none';
        if (feQwen3OptionsDiv) feQwen3OptionsDiv.style.display = 'none';

        if (selectedConfigType === 'OpenAIReasoningLLMConfig' && feOpenAIReasoningOptionsDiv) {
            feOpenAIReasoningOptionsDiv.style.display = 'block';
        } else if (selectedConfigType === 'Qwen3LLMConfig' && feQwen3OptionsDiv) {
            feQwen3OptionsDiv.style.display = 'block';
        }
    }

    function feShowApiSelectionWarning() { if (feLlmApiSelect) feLlmApiSelect.classList.add('input-error'); }
    function feHideApiSelectionWarning() { if (feLlmApiSelect) feLlmApiSelect.classList.remove('input-error'); }

    function feGetLlmConfiguration() {
        if (!feLlmApiSelect) {
            console.error("FrameExtraction: feGetLlmConfiguration: feLlmApiSelect element not found!");
            return { api_type: null };
        }
        const selectedApi = feLlmApiSelect.value;
        const selectedLLMConfigType = feLlmConfigTypeSelect ? feLlmConfigTypeSelect.value : 'BasicLLMConfig';

        const config = {
            api_type: selectedApi,
            llm_config_type: selectedLLMConfigType,
            temperature: parseFloat(temperatureElem?.value) || 0.0,
            max_tokens: parseInt(maxTokensElem?.value) || 512
        };

        if (selectedApi && selectedApi !== "") {
            const optionsDivId = `fe-${selectedApi}-options`;
            const optionsDiv = document.getElementById(optionsDivId);
            if (optionsDiv) {
                optionsDiv.querySelectorAll('input, select, textarea').forEach(el => {
                    if (el.name && el.id) {
                        // The old reasoning model checkboxes (fe-openai-reasoning-model, fe-azure-reasoning-model)
                        // have been removed from HTML, so no specific check needed here for them.
                        // This loop will simply not find them.
                        let key = el.name.startsWith('fe_') ? el.name.substring(3) : el.name; 

                        if (el.type === 'checkbox') {
                            config[key] = el.checked;
                        } else if (el.type === 'number') {
                            const parsedValue = parseFloat(el.value);
                            config[key] = isNaN(parsedValue) ? (el.placeholder ? parseFloat(el.placeholder) : (el.value === "" ? null : el.value)) : parsedValue;
                        } else {
                            config[key] = el.value;
                        }
                    }
                });
            } else {
                console.warn(`FrameExtraction: feGetLlmConfiguration: Could not find options div for ID: ${optionsDivId}.`);
            }
        }
        
        if (selectedLLMConfigType === 'OpenAIReasoningLLMConfig' && feOpenAIReasoningEffortSelect) {
            config.openai_reasoning_effort = feOpenAIReasoningEffortSelect.value;
        } else if (selectedLLMConfigType === 'Qwen3LLMConfig' && feQwenThinkingModeCheckbox) {
            config.qwen_thinking_mode = feQwenThinkingModeCheckbox.checked;
        }
        return config;
    }

    // --- State Management ---
    const feStatePrefix = 'frameExtraction_';
    function saveFrameExtractionState() {
        if (!localStorage) return;
        if (feLlmApiSelect) localStorage.setItem(`${feStatePrefix}llmApiSelectValue`, feLlmApiSelect.value);
        if (feLlmConfigTypeSelect) localStorage.setItem(`${feStatePrefix}llmConfigTypeValue`, feLlmConfigTypeSelect.value);

        allFeApiOptionElements.forEach(el => {
            if (el && el.id) { 
                 if (el.type === 'checkbox') localStorage.setItem(`${feStatePrefix}${el.id}`, el.checked);
                 else localStorage.setItem(`${feStatePrefix}${el.id}`, el.value);
            }
        });
        
        const selectedConfigType = feLlmConfigTypeSelect ? feLlmConfigTypeSelect.value : null;
        if (selectedConfigType === 'OpenAIReasoningLLMConfig' && feOpenAIReasoningEffortSelect) {
            localStorage.setItem(`${feStatePrefix}fe-openai-reasoning-effort`, feOpenAIReasoningEffortSelect.value);
        } else if (selectedConfigType === 'Qwen3LLMConfig' && feQwenThinkingModeCheckbox) {
            localStorage.setItem(`${feStatePrefix}fe-qwen-thinking-mode`, feQwenThinkingModeCheckbox.checked);
        }

        if (inputTextElem) localStorage.setItem(`${feStatePrefix}inputText`, inputTextElem.value);
        if (promptTemplateElem) localStorage.setItem(`${feStatePrefix}promptTemplate`, promptTemplateElem.value);
        if (extractionUnitElem) localStorage.setItem(`${feStatePrefix}extractionUnit`, extractionUnitElem.value);
        if (contextElem) localStorage.setItem(`${feStatePrefix}contextType`, contextElem.value);
        if (temperatureElem) localStorage.setItem(`${feStatePrefix}temperature`, temperatureElem.value);
        if (maxTokensElem) localStorage.setItem(`${feStatePrefix}maxTokens`, maxTokensElem.value);
        if (fuzzyMatchElem) localStorage.setItem(`${feStatePrefix}fuzzyMatch`, fuzzyMatchElem.checked);

        if (outputElem) localStorage.setItem(`${feStatePrefix}extractionOutput`, outputElem.innerHTML);
        if (displayInputElem) localStorage.setItem(`${feStatePrefix}displayInputTextOutput`, displayInputElem.innerHTML);
    }

    function loadFrameExtractionState() {
        if (!localStorage) return;
        const savedApi = localStorage.getItem(`${feStatePrefix}llmApiSelectValue`);
        if (feLlmApiSelect && savedApi) {
            feLlmApiSelect.value = savedApi;
        }
        const savedLlmConfigType = localStorage.getItem(`${feStatePrefix}llmConfigTypeValue`); 
        if (feLlmConfigTypeSelect && savedLlmConfigType) {
            feLlmConfigTypeSelect.value = savedLlmConfigType;
        }
        
        allFeApiOptionElements.forEach(el => {
            if (el && el.id) { 
                const savedValue = localStorage.getItem(`${feStatePrefix}${el.id}`);
                if (savedValue !== null) {
                    if (el.type === 'checkbox') el.checked = (savedValue === 'true');
                    else el.value = savedValue;
                }
            }
        });
        
        const loadedConfigType = feLlmConfigTypeSelect ? feLlmConfigTypeSelect.value : 'BasicLLMConfig';
        if (loadedConfigType === 'OpenAIReasoningLLMConfig' && feOpenAIReasoningEffortSelect) {
            const savedEffort = localStorage.getItem(`${feStatePrefix}fe-openai-reasoning-effort`);
            if (savedEffort !== null) feOpenAIReasoningEffortSelect.value = savedEffort;
        } else if (loadedConfigType === 'Qwen3LLMConfig' && feQwenThinkingModeCheckbox) {
            const savedThinkingMode = localStorage.getItem(`${feStatePrefix}fe-qwen-thinking-mode`);
            if (savedThinkingMode !== null) feQwenThinkingModeCheckbox.checked = (savedThinkingMode === 'true');
        }

        const savedInputText = localStorage.getItem(`${feStatePrefix}inputText`);
        if (inputTextElem && savedInputText) inputTextElem.value = savedInputText;
        const savedPrompt = localStorage.getItem(`${feStatePrefix}promptTemplate`);
        if (promptTemplateElem && savedPrompt) promptTemplateElem.value = savedPrompt;
        
        const savedExtractionUnit = localStorage.getItem(`${feStatePrefix}extractionUnit`);
        if (extractionUnitElem && savedExtractionUnit) extractionUnitElem.value = savedExtractionUnit;
        const savedContextType = localStorage.getItem(`${feStatePrefix}contextType`);
        if (contextElem && savedContextType) contextElem.value = savedContextType;
        
        const savedTemp = localStorage.getItem(`${feStatePrefix}temperature`);
        if (temperatureElem && savedTemp) temperatureElem.value = savedTemp;
        const savedMaxTokens = localStorage.getItem(`${feStatePrefix}maxTokens`);
        if (maxTokensElem && savedMaxTokens) maxTokensElem.value = savedMaxTokens;
        const savedFuzzy = localStorage.getItem(`${feStatePrefix}fuzzyMatch`);
        if (fuzzyMatchElem && savedFuzzy !== null) fuzzyMatchElem.checked = (savedFuzzy === 'true');
        
        // 배치 처리 설정 복원
        const savedBatchEnabled = localStorage.getItem(`${feStatePrefix}fe-enable-batch`);
        if (enableBatchCheckbox && savedBatchEnabled !== null) {
            enableBatchCheckbox.checked = (savedBatchEnabled === 'true');
            // 체크박스 상태에 따라 UI 업데이트
            if (enableBatchCheckbox.checked && batchOptionsDiv) {
                batchOptionsDiv.style.display = 'block';
                startButton.style.display = 'none';
                if (startBatchButton) startBatchButton.style.display = 'inline-block';
            }
        }

        const savedExtractionOutput = localStorage.getItem(`${feStatePrefix}extractionOutput`);
        if (outputElem && savedExtractionOutput) {
            outputElem.innerHTML = savedExtractionOutput;
            const finalResultJsonMatch = savedExtractionOutput.match(/<pre class="final-result-json">(.*?)<\/pre>/s);
            if (finalResultJsonMatch && finalResultJsonMatch[1]) {
                try {
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = finalResultJsonMatch[1];
                    currentExtractedFrames = JSON.parse(tempDiv.textContent || tempDiv.innerText || "");
                    if (downloadButton && currentExtractedFrames && currentExtractedFrames.length > 0) {
                        downloadButton.disabled = false;
                    }
                } catch (e) {
                    console.warn("Could not parse frames from saved output on load:", e);
                    currentExtractedFrames = null;
                    if (downloadButton) downloadButton.disabled = true;
                }
            } else {
                 if (downloadButton) downloadButton.disabled = true;
            }
        } else {
            if (downloadButton) downloadButton.disabled = true;
        }
        const savedDisplayInputTextOutput = localStorage.getItem(`${feStatePrefix}displayInputTextOutput`);
        if (displayInputElem && savedDisplayInputTextOutput) {
            displayInputElem.innerHTML = savedDisplayInputTextOutput;
        }

        setTimeout(() => {
            feUpdateConditionalOptions(); 
            feUpdateConditionalLLMConfigOptions();
        }, 0);
    }

    // --- Event Listeners ---
    if (startButton) {
        startButton.addEventListener('click', async () => {
            const llmConfig = feGetLlmConfiguration();
            if (!llmConfig.api_type) {
                feShowApiSelectionWarning();
                outputElem.innerHTML = '<div class="stream-error-message">Error: Please select an LLM API first.</div>';
                if (downloadButton) downloadButton.disabled = true; 
                currentExtractedFrames = null;
                saveFrameExtractionState(); 
                return;
            }
            feHideApiSelectionWarning();

            const inputText = inputTextElem.value;
            const promptTemplate = promptTemplateElem.value;
            
            // Gemini Direct 처리
            if (llmConfig.api_type === 'gemini_direct') {
                const geminiApiKey = feGeminiApiKey?.value?.trim();
                const geminiModel = feGeminiModel?.value || 'gemini-2.0-flash';
                
                if (!geminiApiKey) {
                    outputElem.innerHTML = '<div class="stream-error-message">Error: Gemini API key is required.</div>';
                    if (downloadButton) downloadButton.disabled = true;
                    return;
                }
                
                await handleGeminiDirectExtraction(inputText, promptTemplate, {
                    apiKey: geminiApiKey,
                    model: geminiModel,
                    temperature: parseFloat(temperatureElem?.value) || 0.2,
                    maxTokens: parseInt(maxTokensElem?.value) || 4096
                });
                return;
            }

            const extractionUnit = extractionUnitElem.value;
            const contextType = contextElem.value;
            const fuzzyMatch = fuzzyMatchElem.checked;

            if (!inputText || !promptTemplate) {
                outputElem.innerHTML = '<div class="stream-error-message">Error: Input text and prompt template are required.</div>';
                if (downloadButton) downloadButton.disabled = true; 
                currentExtractedFrames = null;
                saveFrameExtractionState(); 
                return;
            }

            displayInputElem.innerHTML = escapeHTML(inputText);
            outputElem.innerHTML = 'Starting extraction...\n';
            startButton.disabled = true;
            clearButton.disabled = true;
            if (downloadButton) downloadButton.disabled = true; 
            currentExtractedFrames = null; 
            saveFrameExtractionState();

            let currentUnitId = null;
            let firstChunkForUnit = true;

            const payload = {
                llmConfig: llmConfig,
                inputText: inputText,
                extractorConfig: {
                    prompt_template: promptTemplate,
                    extraction_unit_type: extractionUnit, 
                    context_chunker_type: contextType,   
                    fuzzy_match: fuzzyMatch,
                    allow_overlap_entities: false, 
                    case_sensitive: false         
                }
            };

            fetch('/api/frame-extraction/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        let errorMsg = `HTTP error! status: ${response.status}`;
                        try {
                            const errJson = JSON.parse(text);
                            if (errJson && errJson.error) {
                                errorMsg += ` - ${errJson.error}`;
                            } else {
                                errorMsg += ` - ${text}`;
                            }
                        } catch (e) {
                            errorMsg += ` - ${text}`;
                        }
                        throw new Error(errorMsg);
                    });
                }
                if (!response.body) {
                    throw new Error("ReadableStream not available.");
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let finalFramesTemp = null; 

                function processText({ done, value }) {
                    if (done) {
                        if (finalFramesTemp) { 
                            currentExtractedFrames = finalFramesTemp; 
                            outputElem.innerHTML += `<hr><strong>Extracted Frames:</strong>\n<pre class="final-result-json">${escapeHTML(JSON.stringify(currentExtractedFrames, null, 2))}</pre>`;
                            if (downloadButton && currentExtractedFrames && currentExtractedFrames.length > 0) {
                                downloadButton.disabled = false; 
                            }
                        } else {
                            outputElem.innerHTML += '\n<hr><strong class="info-message">Extraction Finished. No frames extracted or result event not received.</strong>';
                            if (downloadButton) downloadButton.disabled = true;
                            currentExtractedFrames = null;
                        }
                        outputElem.scrollTop = outputElem.scrollHeight;
                        startButton.disabled = false;
                        clearButton.disabled = false;
                        currentUnitId = null;
                        saveFrameExtractionState(); 
                        return;
                    }
                    buffer += decoder.decode(value, { stream: true });
                    let lines = buffer.split('\n');
                    buffer = lines.pop();
                    const oldScrollHeight = outputElem.scrollHeight;
                    const oldScrollTop = outputElem.scrollTop;
                    const clientHeight = outputElem.clientHeight;
                    const wasScrolledToBottom = oldScrollHeight - clientHeight <= oldScrollTop + 10;

                    lines.forEach(line => {
                        if (line.trim() === '') return;

                        if (line.startsWith('data: ')) {
                            const jsonDataString = line.substring(6);
                            if (jsonDataString.trim() === '{}' && lines.some(l => l.startsWith('event: end'))) {
                                return;
                            }
                            try {
                                const json = JSON.parse(jsonDataString);

                                if (outputElem.innerHTML === 'Starting extraction...\n') {
                                    outputElem.innerHTML = '';
                                }

                                switch (json.type) {
                                    case 'info':
                                        outputElem.innerHTML += `<div class="info-message"><strong>INFO:</strong> ${escapeHTML(json.data)}</div>`;
                                        break;
                                    case 'unit':
                                        currentUnitId = json.data.id;
                                        firstChunkForUnit = true;
                                        outputElem.innerHTML += `<hr><div class="unit-block" id="unit-block-${currentUnitId}">`;
                                        outputElem.innerHTML += `<h4 class="unit-header"><strong>UNIT [${escapeHTML(currentUnitId)}] (ID: ${escapeHTML(json.data.id)}, Range: ${escapeHTML(json.data.start)}-${escapeHTML(json.data.end)}):</strong></h4>`;
                                        outputElem.innerHTML += `<div class="unit-processed-text"><div class="text-snippet-header"><strong>Input Text for Unit:</strong></div><pre class="text-snippet-content">${escapeHTML(json.data.text)}</pre></div>`;
                                        outputElem.innerHTML += `<div class="unit-context-container" id="unit-context-${currentUnitId}"></div>`;
                                        outputElem.innerHTML += `<div class="unit-llm-output-container" id="unit-llm-output-${currentUnitId}"><span class="llm-output-header" style="display:none;"><strong>LLM Output:</strong></span><pre class="llm-output-content"></pre></div>`;
                                        outputElem.innerHTML += `</div>`;
                                        break;
                                    case 'context':
                                        if (currentUnitId !== null) {
                                            const contextContainer = document.getElementById(`unit-context-${currentUnitId}`);
                                            if (contextContainer) {
                                                contextContainer.innerHTML = `<div class="unit-context"><strong>Context Provided:</strong><pre>${escapeHTML(json.data)}</pre></div>`;
                                            }
                                        } else {
                                            outputElem.innerHTML += `<div class="general-context"><strong>CONTEXT:</strong><pre>${escapeHTML(json.data)}</pre></div>`;
                                        }
                                        break;
                                    case 'response': 
                                    case 'reasoning': 
                                        if (currentUnitId !== null) {
                                            const llmOutputContainer = document.getElementById(`unit-llm-output-${currentUnitId}`);
                                            if (llmOutputContainer) {
                                                const header = llmOutputContainer.querySelector('.llm-output-header');
                                                const content = llmOutputContainer.querySelector('.llm-output-content');
                                                
                                                if (firstChunkForUnit && header) { // Show header on first actual data chunk for the unit
                                                    header.style.display = 'inline';
                                                    firstChunkForUnit = false;
                                                }
                                                if (content && json.data) { 
                                                    if (json.type === 'reasoning') {
                                                        content.innerHTML += escapeHTML(`*[Reasoning]* ${json.data} `);
                                                    } else { // 'response'
                                                        content.innerHTML += escapeHTML(json.data);
                                                    }
                                                }
                                            }
                                        } else { 
                                            if (json.data) {
                                                if (json.type === 'reasoning') {
                                                    outputElem.innerHTML += escapeHTML(`*[Reasoning]* ${json.data} `);
                                                } else { // 'response'
                                                    outputElem.innerHTML += escapeHTML(json.data);
                                                }
                                            }
                                        }
                                        break;
                                    case 'result':
                                        finalFramesTemp = json.frames; 
                                        break;
                                    case 'error':
                                        outputElem.innerHTML += `<div class="stream-error-message"><strong>STREAM ERROR:</strong> ${escapeHTML(json.message)}</div>`;
                                        if (downloadButton) downloadButton.disabled = true;
                                        currentExtractedFrames = null;
                                        break;
                                    default:
                                        if (jsonDataString.trim() !== '{}') {
                                            console.warn("Received unhandled data type from backend stream:", json);
                                            outputElem.innerHTML += `<div class="unknown-message">UNHANDLED STREAM EVENT (${escapeHTML(json.type || 'undefined')}): ${escapeHTML(json.data !== undefined ? json.data : jsonDataString)}</div>`;
                                        }
                                }
                            } catch (e) {
                                console.error("Failed to parse SSE data:", e, "Line:", line);
                                if (jsonDataString.trim() !== '{}') {
                                    outputElem.innerHTML += `<div class="stream-error-message">Error parsing stream data. See console. (Line: ${escapeHTML(line)})</div>`;
                                }
                            }
                        } else if (line.startsWith('event: end')) {
                            console.log("Client: Received SSE 'event: end' signal.");
                        }
                    });
                    if (wasScrolledToBottom) {
                        outputElem.scrollTop = outputElem.scrollHeight;
                    }
                    reader.read().then(processText).catch(error => {
                        outputElem.innerHTML += `<div class="stream-error-message"><strong>Stream Reading Error:</strong> ${escapeHTML(error.toString())}</div>`;
                        startButton.disabled = false; clearButton.disabled = false; if (downloadButton) downloadButton.disabled = true; currentUnitId = null; currentExtractedFrames = null;
                        saveFrameExtractionState(); 
                    });
                }
                reader.read().then(processText).catch(initialReadError => {
                    outputElem.innerHTML = `<div class="stream-error-message">Error starting stream: ${escapeHTML(initialReadError.toString())}</div>`;
                    startButton.disabled = false; clearButton.disabled = false; if (downloadButton) downloadButton.disabled = true; currentUnitId = null; currentExtractedFrames = null;
                    saveFrameExtractionState();
                });
            })
            .catch(error => {
                outputElem.innerHTML = `<div class="stream-error-message">Error connecting to extraction API: ${escapeHTML(error.toString())}</div>`;
                startButton.disabled = false; clearButton.disabled = false; if (downloadButton) downloadButton.disabled = true; currentUnitId = null; currentExtractedFrames = null;
                saveFrameExtractionState();
            });
        });
    }

    if (clearButton) {
        clearButton.addEventListener('click', () => {
            outputElem.innerHTML = '';
            displayInputElem.innerHTML = ''; 
            startButton.disabled = false;
            clearButton.disabled = false;
            if (downloadButton) downloadButton.disabled = true; 
            currentExtractedFrames = null; 
            localStorage.removeItem(`${feStatePrefix}extractionOutput`);
            localStorage.removeItem(`${feStatePrefix}displayInputTextOutput`);
        });
    }

    if (downloadButton) {
        downloadButton.addEventListener('click', async () => {
            if (!currentExtractedFrames || currentExtractedFrames.length === 0) {
                alert("No frames available to download.");
                return;
            }
            const inputText = inputTextElem.value; 
            if (!inputText) {
                alert("Input text is missing, cannot create a complete .llmie file.");
                return;
            }

            downloadButton.disabled = true;
            downloadButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Downloading...';

            try {
                const response = await fetch('/api/frame-extraction/download', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        inputText: inputText, 
                        frames: currentExtractedFrames
                    })
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const suggestedFilename = response.headers.get('Content-Disposition')?.split('filename=')[1]?.replace(/"/g, '') || 'extraction.llmie';
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = suggestedFilename;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                } else {
                    const errorData = await response.json();
                    alert(`Download failed: ${errorData.error || response.statusText}`);
                }
            } catch (error) {
                console.error("Download error:", error);
                alert(`Download failed: ${error.message}`);
            } finally {
                downloadButton.disabled = false;
                downloadButton.innerHTML = '<i class="fas fa-download"></i>'; 
            }
        });
    }

    if (feLlmApiSelect) {
        feLlmApiSelect.addEventListener('change', () => {
            feUpdateConditionalOptions();
            saveFrameExtractionState();
        });
    }
    if (feLlmConfigTypeSelect) {
        feLlmConfigTypeSelect.addEventListener('change', () => {
            feUpdateConditionalLLMConfigOptions();
            saveFrameExtractionState();
        });
    }

    const feElementsToSaveOnInputOrChange = [
        inputTextElem, promptTemplateElem, 
        extractionUnitElem, contextElem, 
        temperatureElem, maxTokensElem, fuzzyMatchElem,
        ...allFeApiOptionElements,
        feLlmConfigTypeSelect, // The dropdown itself needs to trigger save on change
        feOpenAIReasoningEffortSelect,
        feQwenThinkingModeCheckbox,
        // 배치 처리 요소들 추가
        enableBatchCheckbox,
        batchTypeLlmieRadio,
        batchTypeGeminiRadio,
        apiKeysTextarea,
        chunkSizeInput,
        overlapSizeInput,
        batchSizeInput,
        delayBetweenBatchesInput
    ].filter(el => el); 

    feElementsToSaveOnInputOrChange.forEach(element => {
        if (element) { 
            const eventType = (element.tagName.toLowerCase() === 'textarea' || (element.type && element.type.match(/text|url|password|number|search|email|tel/))) ? 'input' : 'change';
            element.addEventListener(eventType, saveFrameExtractionState);
        }
    });

    // 배치 처리 UI 토글
    console.log('Batch elements check:', {
        enableBatchCheckbox: !!enableBatchCheckbox,
        batchOptionsDiv: !!batchOptionsDiv,
        startBatchButton: !!startBatchButton
    });

    if (enableBatchCheckbox && batchOptionsDiv) {
        console.log('Adding batch checkbox event listener');
        enableBatchCheckbox.addEventListener('change', () => {
            console.log('Batch checkbox changed:', enableBatchCheckbox.checked);
            if (enableBatchCheckbox.checked) {
                batchOptionsDiv.style.display = 'block';
                startButton.style.display = 'none';
                startBatchButton.style.display = 'inline-block';
                console.log('Batch mode enabled');
            } else {
                batchOptionsDiv.style.display = 'none';
                startButton.style.display = 'inline-block';
                startBatchButton.style.display = 'none';
                console.log('Batch mode disabled');
            }
        });
        
        // 배치 체크박스 상태 변경 시에도 저장
        enableBatchCheckbox.addEventListener('change', saveFrameExtractionState);
    } else {
        console.log('Batch elements not found:', {
            enableBatchCheckbox: enableBatchCheckbox,
            batchOptionsDiv: batchOptionsDiv
        });
    }

    // 배치 처리 버튼 이벤트
    if (startBatchButton) {
        console.log('Batch button found, adding event listener');
        startBatchButton.addEventListener('click', async () => {
            console.log('Batch extraction button clicked!');
            const inputText = inputTextElem.value.trim();
            const promptTemplate = promptTemplateElem.value.trim();

            if (!inputText || !promptTemplate) {
                alert('Please provide both input text and prompt template.');
                return;
            }

            // API 키 파싱
            const apiKeysText = apiKeysTextarea.value.trim();
            const apiKeys = apiKeysText.split('\n').map(key => key.trim()).filter(key => key.length > 0);
            
            if (apiKeys.length === 0) {
                alert('Please provide at least one API key for batch processing.');
                return;
            }

            const llmConfig = feGetLlmConfiguration();
            if (!llmConfig.api_type) {
                alert('Please configure the LLM settings properly.');
                return;
            }

            const extractorConfig = {
                prompt_template: promptTemplate,
                extraction_unit_type: extractionUnitElem.value,
                context_chunker_type: contextElem.value,
                slide_window_size: 2, // default value
                case_sensitive: false,
                fuzzy_match: fuzzyMatchElem.checked,
                allow_overlap_entities: false,
                fuzzy_buffer_size: 0.2,
                fuzzy_score_cutoff: 0.8
            };

            // 배치 타입 확인
            const batchType = batchTypeGeminiRadio?.checked ? 'gemini' : 'llmie';
            
            const batchConfig = {
                apiKeys: apiKeys,
                chunkSize: parseInt(chunkSizeInput.value) || 1000,
                overlapSize: parseInt(overlapSizeInput.value) || 100,
                batchSize: parseInt(batchSizeInput.value) || 5,
                delayBetweenBatches: parseFloat(delayBetweenBatchesInput.value) || 2.0,
                geminiModel: 'gemini-2.0-flash',
                temperature: parseFloat(temperatureElem?.value) || 0.2,
                maxTokens: parseInt(maxTokensElem?.value) || 4096
            };

            // UI 상태 업데이트
            startBatchButton.disabled = true;
            clearButton.disabled = true;
            if (downloadButton) downloadButton.disabled = true;
            currentExtractedFrames = null;

            displayInputElem.innerHTML = escapeHTML(inputText);
            
            if (batchType === 'gemini') {
                outputElem.innerHTML = '<div class="stream-info-message">Starting Gemini Direct API batch processing...</div>';
            } else {
                outputElem.innerHTML = '<div class="stream-info-message">Starting LLM-IE framework batch processing...</div>';
            }

            let payload, endpoint;
            
            if (batchType === 'gemini') {
                // Gemini 직접 호출용 페이로드
                payload = {
                    inputText: inputText,
                    promptTemplate: promptTemplate,
                    batchConfig: batchConfig
                };
                endpoint = '/api/frame-extraction/gemini-batch';
            } else {
                // LLM-IE 프레임워크용 페이로드
                payload = {
                    inputText: inputText,
                    llmConfig: llmConfig,
                    extractorConfig: extractorConfig,
                    batchConfig: batchConfig
                };
                endpoint = '/api/frame-extraction/batch';
            }

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const reader = response.body.getReader();
                let buffer = '';
                let allResults = [];

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += new TextDecoder().decode(value);
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // Keep incomplete line in buffer

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const eventData = JSON.parse(line.slice(6));
                                handleBatchEvent(eventData, allResults);
                            } catch (e) {
                                console.error('Failed to parse batch event:', e);
                                console.error('Raw line:', line);
                                console.error('Line slice:', line.slice(6));
                            }
                        }
                    }
                }
            } catch (error) {
                outputElem.innerHTML = `<div class="stream-error-message">Batch processing error: ${escapeHTML(error.toString())}</div>`;
                console.error('Batch processing error:', error);
            } finally {
                startBatchButton.disabled = false;
                clearButton.disabled = false;
                saveFrameExtractionState();
            }
        });
    }

    async function handleGeminiDirectExtraction(inputText, promptTemplate, geminiConfig) {
        displayInputElem.innerHTML = escapeHTML(inputText);
        outputElem.innerHTML = '<div class="stream-info-message">Starting Gemini Direct extraction...</div>';
        startButton.disabled = true;
        clearButton.disabled = true;
        if (downloadButton) downloadButton.disabled = true;
        currentExtractedFrames = null;

        const payload = {
            inputText: inputText,
            promptTemplate: promptTemplate,
            geminiConfig: geminiConfig
        };

        try {
            const response = await fetch('/api/frame-extraction/gemini-single', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const reader = response.body.getReader();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += new TextDecoder().decode(value);
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const eventData = JSON.parse(line.slice(6));
                            handleSingleExtractionEvent(eventData);
                        } catch (e) {
                            console.warn('Failed to parse Gemini event:', e);
                        }
                    }
                }
            }
        } catch (error) {
            outputElem.innerHTML += `<div class="stream-error-message">Gemini extraction error: ${escapeHTML(error.toString())}</div>`;
            console.error('Gemini extraction error:', error);
        } finally {
            startButton.disabled = false;
            clearButton.disabled = false;
            saveFrameExtractionState();
        }
    }

    function handleSingleExtractionEvent(eventData) {
        const { type, data, frames, message, document } = eventData;
        
        switch (type) {
            case 'info':
                // 로그 메시지 숨김 - 콘솔에만 출력
                console.log('[INFO]', data);
                break;
            case 'warning':
                // 경고는 표시
                outputElem.innerHTML += `<div class="stream-warning-message">${data}</div>`;
                break;
            case 'debug':
                // 디버그 메시지 숨김 - 콘솔에만 출력
                console.log('[DEBUG]', data);
                break;
            case 'result':
                if (frames && frames.length > 0) {
                    currentExtractedFrames = frames;
                    
                    // LLM-IE 문서 형태로 결과 표시
                    const llmieDocument = document || {
                        doc_id: 'extraction_result',
                        text: inputTextElem.value,
                        frames: frames,
                        relations: []
                    };
                    
                    // JSON 형태로 결과 표시
                    const resultJson = JSON.stringify(llmieDocument, null, 2);
                    outputElem.innerHTML += `<div class="extraction-result">
                        <h3>Extraction Results (${frames.length} frames)</h3>
                        <pre class="llmie-result">${escapeHTML(resultJson)}</pre>
                    </div>`;
                    
                    // 다운로드 버튼 활성화
                    if (downloadButton) {
                        downloadButton.disabled = false;
                    }
                } else {
                    outputElem.innerHTML += `<div class="stream-warning-message">No frames extracted</div>`;
                }
                break;
            case 'error':
                outputElem.innerHTML += `<div class="stream-error-message">Error: ${message}</div>`;
                break;
        }
        
        outputElem.scrollTop = outputElem.scrollHeight;
    }

    function handleBatchEvent(eventData, allResults) {
        try {
            const { type, data } = eventData;
            
            // 디버깅을 위해 이벤트 데이터 상세 로그
            console.log('[EVENT]', type);
            console.log('[EVENT DATA]', JSON.stringify(eventData, null, 2));
            
            // 다운로드 버튼 상태 확인
            if (downloadButton) {
                console.log('[DOWNLOAD BTN]', downloadButton.disabled ? 'DISABLED' : 'ENABLED');
                console.log('[CURRENT FRAMES]', currentExtractedFrames ? currentExtractedFrames.length : 'NONE');
            }
            
            switch (type) {
            case 'batch_start':
                // 로그를 콘솔로만 출력
                console.log(`[BATCH] Starting batch ${data.batch_number}/${data.total_batches} (${data.batch_size} chunks) - Progress: ${data.progress}`);
                break;
                
            case 'chunk_start':
                // 청크 시작 로그 숨김
                console.log(`[CHUNK] Processing chunk ${data.chunk_index + 1}...`);
                break;
                
            case 'chunk_complete':
                allResults.push(data);
                const framesInfo = data.frames_count ? ` (${data.frames_count} frames)` : '';
                console.log(`[CHUNK] Completed chunk ${data.chunk_index}${framesInfo}`);
                break;
                
            case 'chunk_error':
                outputElem.innerHTML += `<div class="stream-error-message">✗ Error in chunk ${data.chunk_index + 1}: ${data.error}</div>`;
                break;
                
            case 'batch_complete':
                console.log(`[BATCH] Completed batch ${data.batch_number} - Progress: ${data.progress}`);
                break;
                
            case 'processing_complete':
                // 간단한 완료 메시지만 표시
                outputElem.innerHTML += `<div class="stream-success-message">✅ Processing Complete: ${data.total_frames || 0} frames extracted</div>`;
                console.log(`[BATCH] Processing complete! ${data.total_processed} chunks processed`);
                console.log(`[BATCH] API Key Usage:`, data.api_key_usage);
                
                // 상세한 데이터 구조 디버깅
                console.log('[PROCESSING_COMPLETE] data keys:', Object.keys(data));
                console.log('[PROCESSING_COMPLETE] all_frames type:', typeof data.all_frames);
                console.log('[PROCESSING_COMPLETE] all_frames length:', data.all_frames ? data.all_frames.length : 'UNDEFINED');
                if (data.all_frames && data.all_frames.length > 0) {
                    console.log('[PROCESSING_COMPLETE] First frame sample:', data.all_frames[0]);
                }
                
                // 모든 프레임이 있다면 결과를 표시하고 다운로드 버튼 활성화
                if (data.all_frames && data.all_frames.length > 0) {
                    console.log('[SUCCESS] Found frames in processing_complete:', data.all_frames.length);
                    currentExtractedFrames = data.all_frames;
                    
                    // LLM-IE 문서 구조 생성
                    const llmieDocument = {
                        doc_id: 'batch_extraction_result',
                        text: inputTextElem.value,
                        frames: data.all_frames,
                        relations: []
                    };
                    
                    // JSON 형태로 결과 표시
                    const resultJson = JSON.stringify(llmieDocument, null, 2);
                    outputElem.innerHTML += `<div class="extraction-result">
                        <h3>Batch Extraction Results (${data.all_frames.length} frames)</h3>
                        <pre class="llmie-result">${escapeHTML(resultJson)}</pre>
                    </div>`;
                    
                    // 다운로드 버튼 활성화
                    if (downloadButton) {
                        downloadButton.disabled = false;
                        console.log('✅ Download button activated from processing_complete with', data.all_frames.length, 'frames');
                    }
                } else {
                    console.log('[ERROR] No frames found in processing_complete event!');
                    console.log('[ERROR] data.all_frames:', data.all_frames);
                }
                break;
                
            case 'result':
                // LLM-IE 문서 형태로 결과 처리
                // 서버에서 보내는 구조: {'type': 'result', 'frames': all_frames, 'document': llm_ie_document}
                console.log('[RESULT EVENT] eventData:', eventData);
                console.log('[RESULT EVENT] eventData.frames:', eventData.frames);
                console.log('[RESULT EVENT] eventData.document:', eventData.document);
                
                const frames = eventData.frames || eventData.data?.frames;
                const document = eventData.document || eventData.data?.document;
                
                console.log('[RESULT EVENT] extracted frames:', frames);
                console.log('[RESULT EVENT] extracted document:', document);
                
                if (frames && frames.length > 0) {
                    currentExtractedFrames = frames;
                    
                    // LLM-IE 문서 구조 생성 (document가 있으면 사용, 없으면 생성)
                    const llmieDocument = document || {
                        doc_id: 'batch_extraction_result',
                        text: inputTextElem.value,
                        frames: frames,
                        relations: []
                    };
                    
                    // JSON 형태로 결과 표시
                    const resultJson = JSON.stringify(llmieDocument, null, 2);
                    outputElem.innerHTML += `<div class="extraction-result">
                        <h3>Batch Extraction Results (${frames.length} frames)</h3>
                        <pre class="llmie-result">${escapeHTML(resultJson)}</pre>
                    </div>`;
                    
                    // 다운로드 버튼 활성화
                    if (downloadButton) {
                        downloadButton.disabled = false;
                        console.log('✅ Download button activated with', frames.length, 'frames');
                    }
                } else {
                    outputElem.innerHTML += `<div class="stream-warning-message">No frames extracted from the text</div>`;
                    console.log('⚠️ No frames found in result event');
                    console.log('[RESULT EVENT] eventData structure:', JSON.stringify(eventData, null, 2));
                }
                break;
                
            case 'batch_processing_complete':
                // 호환성을 위해 유지
                outputElem.innerHTML += `<div class="stream-success-message">🎉 Batch processing complete! Processed ${data.total_chunks_processed || data.total_processed || 0} chunks</div>`;
                outputElem.innerHTML += `<div class="stream-info-message">API Key Usage: ${JSON.stringify(data.api_key_usage, null, 2)}</div>`;
                break;
                
            case 'batch_error':
                outputElem.innerHTML += `<div class="stream-error-message">Batch processing failed: ${data.message}</div>`;
                break;
        }
        
        // 스크롤을 맨 아래로
        outputElem.scrollTop = outputElem.scrollHeight;
        } catch (error) {
            console.error('[BATCH EVENT ERROR]', error);
            console.error('[BATCH EVENT ERROR] eventData:', eventData);
            console.error('[BATCH EVENT ERROR] type:', eventData?.type);
            outputElem.innerHTML += `<div class="stream-error-message">Event handling error: ${error.message}</div>`;
        }
    }

    loadFrameExtractionState(); 
});