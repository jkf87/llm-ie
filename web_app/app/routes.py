# app/routes.py
# This file defines the web application routes.

import os
import json
import time
import tempfile
import traceback
from flask import (
    Blueprint, render_template, request, Response, jsonify,
    stream_with_context, current_app, send_file
)

# Services from app_services.py
from .app_services import (
    create_llm_engine_from_config,
    get_app_frame_extractor
)
from .batch_services import BatchProcessor, split_text_into_chunks
from .gemini_direct_engine import GeminiDirectBatchProcessor
from .knowledge_graph_agents import KnowledgeGraphGenerator, KnowledgeGraphValidator

# Data types from your llm_ie library (if needed directly in routes, otherwise services handle them)
from llm_ie.data_types import LLMInformationExtractionDocument, LLMInformationExtractionFrame
from llm_ie.prompt_editor import PromptEditor
from llm_ie.engines import BasicLLMConfig, OpenAIReasoningLLMConfig, Qwen3LLMConfig
from llm_ie.extractors import DirectFrameExtractor # For PromptEditor type hint

# LLM API Options to pass to the template (could also be managed in app/__init__.py)
LLM_API_OPTIONS = [
    {"value": "gemini_direct", "name": "Gemini Direct API (Recommended)"},
    {"value": "openai_compatible", "name": "OpenAI Compatible"},
    {"value": "ollama", "name": "Ollama"},
    {"value": "huggingface_hub", "name": "HuggingFace Hub"},
    {"value": "openai", "name": "OpenAI"},
    {"value": "azure_openai", "name": "Azure OpenAI"},
    {"value": "litellm", "name": "LiteLLM"},
]

# Create a Blueprint
main_bp = Blueprint('main', __name__, template_folder='templates', static_folder='static')

@main_bp.route('/')
def application_shell():
    """
    Serves the main application shell.
    """
    return render_template(
        'app_shell.html',
        llm_api_options=LLM_API_OPTIONS,
        active_tab='prompt-editor' # Default active tab
    )

@main_bp.route('/api/prompt-editor/chat', methods=['POST'])
def api_prompt_editor_chat():
    data = request.json
    messages = data.get('messages', [])
    llm_config_from_request = data.get('llmConfig', {})

    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    if not llm_config_from_request or not llm_config_from_request.get('api_type'):
        return jsonify({"error": "LLM API configuration ('api_type') is missing"}), 400
    
    try:
        # create_llm_engine_from_config now handles temperature and max_tokens via LLMConfig
        engine_to_use = create_llm_engine_from_config(llm_config_from_request)
        current_app.logger.info(f"PromptEditor: Successfully created engine: {type(engine_to_use).__name__}")
        # DirectFrameExtractor is a placeholder for the type expected by PromptEditor
        editor = PromptEditor(engine_to_use, DirectFrameExtractor) 

        def generate_chat_stream():
            try:
                stream = editor.chat_stream(
                    messages=messages
                )
                for chunk in stream:
                    yield f"data: {json.dumps(chunk)}\n\n"


            except Exception as e:
                current_app.logger.error(f"Error during PromptEditor stream generation: {e}\n{traceback.format_exc()}")
                yield f"data: {json.dumps({'error': f'Stream generation failed: {type(e).__name__} - {str(e)}'})}\n\n"
            finally:
                yield "event: end\ndata: {}\n\n" 

        return Response(stream_with_context(generate_chat_stream()), mimetype='text/event-stream')

    except (ValueError, ImportError, RuntimeError, TypeError) as e:
        current_app.logger.error(f"Failed to create LLM engine or process PromptEditor request: {e}")
        status_code = 400 if isinstance(e, (ValueError, TypeError)) else 500
        return jsonify({"error": f"{str(e)}"}), status_code
    except Exception as e:
        current_app.logger.error(f"Unexpected error in /api/prompt-editor/chat: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500


@main_bp.route('/api/frame-extraction/stream', methods=['POST'])
def api_frame_extraction_stream():
    """
    Handles requests for frame extraction.
    Streams the extraction process and results.
    """
    data = request.json
    input_text = data.get('inputText', '')
    llm_config_from_request = data.get('llmConfig', {})
    extractor_config_req = data.get('extractorConfig', {})

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400
    if not extractor_config_req.get('prompt_template'):
        return jsonify({"error": "Prompt template is required in extractorConfig"}), 400
    if not llm_config_from_request or not llm_config_from_request.get('api_type'):
        return jsonify({"error": "LLM API configuration ('api_type') is missing"}), 400

    try:
        # create_llm_engine_from_config now handles temperature and max_tokens via LLMConfig
        engine_to_use = create_llm_engine_from_config(llm_config_from_request)
        current_app.logger.info(f"Frame Extraction: Created engine: {type(engine_to_use).__name__}")

        extractor = get_app_frame_extractor(engine_to_use, extractor_config_req)
        current_app.logger.info(f"Frame Extraction: Created extractor: {type(extractor).__name__}")

        def generate_extraction_stream():
            all_extraction_unit_results = []
            try:
                # Temperature and max_new_tokens are now part of the engine's config
                # qwen_no_think is also removed as it's not a standard InferenceEngine.chat param
                # If Qwen specific logic is needed, it should be in Qwen3LLMConfig.preprocess_messages

                current_app.logger.debug(f"Calling AppDirectFrameExtractor.stream...")
                # The `stream` method in `AppDirectFrameExtractor` itself calls `InferenceEngine.chat`
                # which now uses the LLMConfig for temperature and max_new_tokens.
                # So, we don't pass them here.
                stream_generator = extractor.stream(
                    text_content=input_text,
                    document_key=None # Assuming inputText is the direct document
                    # temperature, max_new_tokens, qwen_no_think removed from here
                )
                # ... (the rest of the stream processing logic for events remains the same) ...
                while True:
                    try:
                        event = next(stream_generator) # This will yield dicts like {"type": "info", "data": ...}
                        yield f"data: {json.dumps(event)}\n\n"
                    except StopIteration as e: 
                        all_extraction_unit_results = e.value # This is `collected_results` from your extractor's stream method
                        if not isinstance(all_extraction_unit_results, list):
                            current_app.logger.warning(f"Extractor.stream() did not return a list. Got: {type(all_extraction_unit_results)}. Assuming empty.")
                            all_extraction_unit_results = []
                        break 

                current_app.logger.info(f"Frame Extraction: Stream finished. Collected {len(all_extraction_unit_results)} unit results.")

                post_process_params = { # ... (remains the same)
                    "case_sensitive": extractor_config_req.get('case_sensitive', False),
                    "fuzzy_match": extractor_config_req.get('fuzzy_match', True),
                    "allow_overlap_entities": extractor_config_req.get('allow_overlap_entities', False),
                    "fuzzy_buffer_size": float(extractor_config_req.get('fuzzy_buffer_size', 0.2)),
                    "fuzzy_score_cutoff": float(extractor_config_req.get('fuzzy_score_cutoff', 0.8)),
                }
                current_app.logger.debug(f"Calling extractor.post_process_frames with params: {post_process_params}")

                final_frames = extractor.post_process_frames(
                    extraction_results=all_extraction_unit_results,
                    **post_process_params
                )
                frames_dict_list = [f.to_dict() for f in final_frames]
                yield f"data: {json.dumps({'type': 'result', 'frames': frames_dict_list})}\n\n"
                current_app.logger.info(f"Frame Extraction: Post-processing complete, {len(final_frames)} frames found.")

            except Exception as e: # ... (error handling in stream)
                current_app.logger.error(f"Error during frame extraction stream/processing: {e}\n{traceback.format_exc()}")
                error_payload = {'type': 'error', 'message': f'Extraction failed: {type(e).__name__} - {str(e)}'}
                yield f"data: {json.dumps(error_payload)}\n\n"
            finally:
                yield "event: end\ndata: {}\n\n" 

        return Response(stream_with_context(generate_extraction_stream()), mimetype='text/event-stream')

    except (ValueError, ImportError, RuntimeError, TypeError) as e:
        current_app.logger.error(f"Failed to create engine/extractor or process FrameExtraction request: {e}")
        status_code = 400 if isinstance(e, (ValueError, TypeError)) else 500
        return jsonify({"error": f"{str(e)}"}), status_code
    except Exception as e:
        current_app.logger.error(f"Unexpected error in /api/frame-extraction/stream setup: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred during setup."}), 500


@main_bp.route('/api/frame-extraction/download', methods=['POST'])
def api_frame_extraction_download():
    """
    Handles requests to download extracted frames as an .llmie file.
    """
    data = request.json
    input_text = data.get('inputText')
    frames_data_dicts = data.get('frames') # Expects a list of frame dictionaries

    if input_text is None or frames_data_dicts is None:
        return jsonify({"error": "Missing 'inputText' or 'frames' in request"}), 400

    tmp_file_path = None
    try:
        frame_objects = []
        if isinstance(frames_data_dicts, list):
            for frame_dict in frames_data_dicts:
                # Basic validation for essential keys
                if not all(k in frame_dict for k in ['frame_id', 'start', 'end', 'entity_text']):
                    current_app.logger.warning(f"Skipping frame dict due to missing keys for download: {frame_dict}")
                    continue
                frame_objects.append(LLMInformationExtractionFrame.from_dict(frame_dict))

        # Create an LLMInformationExtractionDocument instance
        doc = LLMInformationExtractionDocument(doc_id="downloaded_extraction", text=input_text)
        doc.add_frames(frame_objects, create_id=False) # Assuming IDs are already present from client

        # Save to a temporary file
        # delete=False is important because send_file needs the file to exist after this block
        with tempfile.NamedTemporaryFile(suffix=".llmie", delete=False, mode='w', encoding='utf-8') as tmp_file:
            tmp_file_path = tmp_file.name
            doc.save(tmp_file_path) # LLMInformationExtractionDocument.save method

        timestamp = current_app.config.get("CURRENT_TIME", "timestamp") # Get time from app config or default
        filename = f"extraction_results_{timestamp}.llmie"

        # send_file will delete the temp file after sending if `tmp_file.delete=True` was used,
        # but since we used delete=False, we need to manage it in `finally`.
        # However, it's safer to let Flask handle it if possible, or clean up robustly.
        # For now, we rely on manual deletion in the finally block.
        return send_file(
            tmp_file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
    except Exception as e:
        current_app.logger.error(f"Error during download file preparation: {e}", exc_info=True)
        return jsonify({"error": f"Failed to prepare download file: {str(e)}"}), 500
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
                current_app.logger.info(f"Temporary download file {tmp_file_path} deleted.")
            except Exception as e_remove:
                current_app.logger.error(f"Error deleting temporary download file {tmp_file_path}: {e_remove}")


@main_bp.route('/api/results/process_llmie_data', methods=['POST'])
def api_results_process_llmie_data():
    """
    Processes an uploaded .llmie file and returns its content.
    """
    if 'llmie_file' not in request.files:
        current_app.logger.error("No 'llmie_file' part in the request.")
        return jsonify({"error": "No .llmie file part in the request"}), 400

    file_storage = request.files['llmie_file']

    if file_storage.filename == '':
        current_app.logger.error("No file selected for upload.")
        return jsonify({"error": "No .llmie file selected"}), 400

    if not file_storage.filename.endswith('.llmie'):
        current_app.logger.error(f"Invalid file type uploaded: {file_storage.filename}")
        return jsonify({"error": "Invalid file type. Please upload an .llmie file"}), 400

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".llmie") as tmp_file:
            file_storage.save(tmp_file) # Save the uploaded data to this temp file
            temp_file_path = tmp_file.name
        current_app.logger.info(f"Uploaded .llmie file saved temporarily to: {temp_file_path}")

        doc = LLMInformationExtractionDocument(filename=temp_file_path) # Load from the temp file
        current_app.logger.info(f"Successfully loaded .llmie document: {doc.doc_id if doc.doc_id else 'No ID'}")

        text_content = doc.text
        frames_list = [frame.to_dict() for frame in doc.frames]
        # Assuming relations in LLMInformationExtractionDocument are already dicts or have a to_dict()
        relations_list = doc.relations if doc.relations else []


        attribute_keys_set = set()
        if doc.frames:
            for frame in doc.frames:
                if hasattr(frame, 'attr') and isinstance(frame.attr, dict):
                    for key in frame.attr.keys():
                        attribute_keys_set.add(key)
        sorted_attribute_keys = sorted(list(attribute_keys_set))
        current_app.logger.info(f"Extracted attribute keys: {sorted_attribute_keys}")

        return jsonify({
            "text": text_content,
            "frames": frames_list,
            "relations": relations_list,
            "attribute_keys": sorted_attribute_keys
        })
    except Exception as e:
        current_app.logger.error(f"Error processing .llmie file: {e}", exc_info=True)
        return jsonify({"error": f"Failed to process .llmie file: {str(e)}"}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                current_app.logger.info(f"Temporary .llmie file {temp_file_path} deleted.")
            except Exception as e_remove:
                current_app.logger.error(f"Error deleting temporary .llmie file {temp_file_path}: {e_remove}")


@main_bp.route('/api/results/render', methods=['POST'])
def api_results_render():
    """
    Renders visualization HTML from provided text, frames, and relations data.
    """
    data = request.json
    text_content = data.get('text', "")
    frames_data_dicts = data.get('frames', [])
    relations_data_dicts = data.get('relations', []) # Assuming this is already a list of dicts
    viz_options = data.get('vizOptions', {})
    color_attr_key = viz_options.get('color_attr_key') # Can be None or empty string

    try:
        if color_attr_key:
            current_app.logger.info(f"Rendering visualization with color_attr_key: '{color_attr_key}'")
        else:
            current_app.logger.info("Rendering visualization without color_attr_key")

        doc = LLMInformationExtractionDocument(doc_id="viz_doc_from_data", text=text_content)

        if frames_data_dicts:
            frames_to_add = []
            for frame_dict in frames_data_dicts:
                if all(k in frame_dict for k in ['frame_id', 'start', 'end', 'entity_text']):
                    frames_to_add.append(LLMInformationExtractionFrame.from_dict(frame_dict))
                else:
                    current_app.logger.warning(f"Skipping invalid frame dictionary for visualization: {frame_dict}")
            if frames_to_add:
                doc.add_frames(frames_to_add, create_id=False) # Assume IDs are final

        if relations_data_dicts: # relations_data_dicts should be a list of dicts
            doc.add_relations(relations_data_dicts)

        # Call viz_render from the LLMInformationExtractionDocument instance
        html_content = doc.viz_render(
            theme='light', # Or make this configurable
            color_attr_key=color_attr_key if color_attr_key else None # Pass None if empty string
        )
        return jsonify({"html": html_content})

    except ImportError as ie:
        current_app.logger.error(f"Import Error in rendering: {ie}")
        return jsonify({"error": f"Visualization library (e.g., ie-viz) not installed correctly: {ie}"}), 500
    except Exception as e:
        current_app.logger.error(f"Error in /api/results/render: {e}", exc_info=True)
        return jsonify({"error": f"Failed to render visualization: {str(e)}"}), 500


@main_bp.route('/api/frame-extraction/batch', methods=['POST'])
def api_frame_extraction_batch():
    """
    배치 처리를 위한 엔드포인트 - 여러 API 키와 청크 기반 처리
    """
    data = request.json
    input_text = data.get('inputText', '')
    llm_config_from_request = data.get('llmConfig', {})
    extractor_config_req = data.get('extractorConfig', {})
    batch_config = data.get('batchConfig', {})
    
    # 배치 설정 파라미터
    api_keys = batch_config.get('apiKeys', [])  # 여러 API 키 리스트
    chunk_size = int(batch_config.get('chunkSize', 1000))
    overlap_size = int(batch_config.get('overlapSize', 100))
    batch_size = int(batch_config.get('batchSize', 5))
    delay_between_batches = float(batch_config.get('delayBetweenBatches', 2.0))
    
    if not input_text:
        return jsonify({"error": "Input text is required"}), 400
    if not extractor_config_req.get('prompt_template'):
        return jsonify({"error": "Prompt template is required in extractorConfig"}), 400
    if not llm_config_from_request or not llm_config_from_request.get('api_type'):
        return jsonify({"error": "LLM API configuration ('api_type') is missing"}), 400
    if not api_keys or len(api_keys) == 0:
        return jsonify({"error": "At least one API key is required in batchConfig.apiKeys"}), 400
    
    try:
        # 텍스트를 청크로 분할
        text_chunks = split_text_into_chunks(input_text, chunk_size, overlap_size)
        current_app.logger.info(f"Split text into {len(text_chunks)} chunks (size: {chunk_size}, overlap: {overlap_size})")
        
        # 배치 프로세서 초기화
        batch_processor = BatchProcessor(llm_config_from_request, api_keys)
        
        def generate_batch_stream():
            try:
                stream_generator = batch_processor.process_chunks_batch(
                    text_chunks=text_chunks,
                    extractor_config=extractor_config_req,
                    batch_size=batch_size,
                    delay_between_batches=delay_between_batches
                )
                
                all_results = []
                for event in stream_generator:
                    # 스트리밍으로 진행상황 전송
                    yield f"data: {json.dumps(event)}\n\n"
                    
                    # 완료된 청크 결과 수집
                    if event.get('type') == 'chunk_complete':
                        all_results.append(event['data'])
                    elif event.get('type') == 'processing_complete':
                        # 전체 처리 완료 이벤트 전송
                        final_event = {
                            'type': 'batch_processing_complete',
                            'data': {
                                'total_chunks_processed': len(all_results),
                                'all_results': all_results,
                                'api_key_usage': event['data'].get('api_key_usage', {})
                            }
                        }
                        yield f"data: {json.dumps(final_event)}\n\n"
                        
            except Exception as e:
                current_app.logger.error(f"Error during batch processing: {e}\n{traceback.format_exc()}")
                error_payload = {'type': 'batch_error', 'message': f'Batch processing failed: {type(e).__name__} - {str(e)}'}
                yield f"data: {json.dumps(error_payload)}\n\n"
            finally:
                yield "event: end\ndata: {}\n\n"
        
        return Response(stream_with_context(generate_batch_stream()), mimetype='text/event-stream')
        
    except Exception as e:
        current_app.logger.error(f"Failed to setup batch processing: {e}")
        return jsonify({"error": f"Batch processing setup failed: {str(e)}"}), 500


@main_bp.route('/api/frame-extraction/gemini-batch', methods=['POST'])
def api_frame_extraction_gemini_batch():
    """
    Gemini 직접 호출을 사용한 배치 처리 엔드포인트
    """
    data = request.json
    input_text = data.get('inputText', '')
    prompt_template = data.get('promptTemplate', '')
    batch_config = data.get('batchConfig', {})
    
    # 배치 설정 파라미터
    api_keys = batch_config.get('apiKeys', [])
    chunk_size = int(batch_config.get('chunkSize', 1000))
    overlap_size = int(batch_config.get('overlapSize', 100))
    batch_size = int(batch_config.get('batchSize', 5))
    delay_between_batches = float(batch_config.get('delayBetweenBatches', 2.0))
    gemini_model = batch_config.get('geminiModel', 'gemini-2.0-flash')
    temperature = float(batch_config.get('temperature', 0.2))
    max_tokens = int(batch_config.get('maxTokens', 4096))
    
    if not input_text:
        return jsonify({"error": "Input text is required"}), 400
    if not prompt_template:
        return jsonify({"error": "Prompt template is required"}), 400
    if not api_keys or len(api_keys) == 0:
        return jsonify({"error": "At least one API key is required"}), 400
    
    try:
        # 텍스트를 청크로 분할 (청크와 시작 위치를 함께 받음)
        chunk_data = split_text_into_chunks(input_text, chunk_size, overlap_size)
        current_app.logger.info(f"Split text into {len(chunk_data)} chunks (size: {chunk_size}, overlap: {overlap_size})")
        
        # Gemini 배치 프로세서 초기화
        gemini_config = {
            'gemini_model': gemini_model,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        batch_processor = GeminiDirectBatchProcessor(api_keys, gemini_config)
        
        def generate_gemini_batch_stream():
            try:
                total_chunks = len(chunk_data)
                processed_chunks = 0
                all_results = []
                current_frame_id_offset = 0  # 고유 frame_id를 위한 오프셋 추적
                
                # 배치 단위로 처리
                for batch_start in range(0, total_chunks, batch_size):
                    batch_end = min(batch_start + batch_size, total_chunks)
                    batch_chunk_data = chunk_data[batch_start:batch_end]
                    batch_number = (batch_start // batch_size) + 1
                    total_batches = (total_chunks + batch_size - 1) // batch_size
                    
                    yield f"data: {json.dumps({'type': 'batch_start', 'data': {'batch_number': batch_number, 'total_batches': total_batches, 'batch_size': len(batch_chunk_data), 'progress': f'{processed_chunks}/{total_chunks}'}})}\n\n"
                    
                    # 배치 내 청크들 처리
                    batch_results = []
                    for i, (chunk, chunk_start_pos) in enumerate(batch_chunk_data):
                        chunk_index = batch_start + i
                        
                        yield f"data: {json.dumps({'type': 'chunk_start', 'data': {'chunk_index': chunk_index + 1, 'chunk_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk}})}\n\n"
                        
                        try:
                            # 청크 처리 - 고유한 frame_id_offset 및 시작 위치 전달
                            result = batch_processor.process_chunk(chunk, prompt_template, current_frame_id_offset, chunk_start_pos)
                            
                            if result.get('success'):
                                extracted_frames = result.get('extracted_frames', [])
                                batch_results.append({
                                    'chunk_index': chunk_index,
                                    'raw_content': result.get('raw_content', ''),
                                    'extracted_frames': extracted_frames,
                                    'chunk_preview': result.get('chunk_preview', ''),
                                    'status': 'success'
                                })
                                
                                # frame_id_offset 업데이트
                                current_frame_id_offset += len(extracted_frames)
                                
                                yield f"data: {json.dumps({'type': 'chunk_complete', 'data': {'chunk_index': chunk_index + 1, 'frames_count': len(extracted_frames), 'preview': result.get('chunk_preview', ''), 'raw_content_preview': result.get('raw_content', '')[:200] + '...' if len(result.get('raw_content', '')) > 200 else result.get('raw_content', '')}})}\n\n"
                            else:
                                batch_results.append({
                                    'chunk_index': chunk_index,
                                    'error': result.get('error', 'Unknown error'),
                                    'chunk_preview': result.get('chunk_preview', ''),
                                    'status': 'error'
                                })
                                
                                yield f"data: {json.dumps({'type': 'chunk_error', 'data': {'chunk_index': chunk_index + 1, 'error': result.get('error', 'Unknown error')}})}\n\n"
                            
                        except Exception as e:
                            batch_results.append({
                                'chunk_index': chunk_index,
                                'error': str(e),
                                'status': 'error'
                            })
                            
                            yield f"data: {json.dumps({'type': 'chunk_error', 'data': {'chunk_index': chunk_index + 1, 'error': str(e)}})}\n\n"
                        
                        processed_chunks += 1
                    
                    all_results.extend(batch_results)
                    yield f"data: {json.dumps({'type': 'batch_complete', 'data': {'batch_number': batch_number, 'results': batch_results, 'progress': f'{processed_chunks}/{total_chunks}'}})}\n\n"
                    
                    # 배치 간 대기
                    if batch_end < total_chunks and delay_between_batches > 0:
                        current_app.logger.info(f"Waiting {delay_between_batches}s before next batch...")
                        time.sleep(delay_between_batches)
                
                # 모든 프레임 수집
                all_frames = []
                for batch_result in all_results:
                    if batch_result.get('status') == 'success' and batch_result.get('extracted_frames'):
                        all_frames.extend(batch_result['extracted_frames'])
                
                # 완료 이벤트 - 실제 추출 결과 포함
                final_data = {
                    'total_processed': processed_chunks,
                    'total_frames': len(all_frames),
                    'all_frames': all_frames,
                    'api_key_usage': dict(batch_processor.key_usage_count),
                    'batch_results_summary': [
                        {
                            'chunk_index': r.get('chunk_index'),
                            'status': r.get('status'),
                            'frames_count': len(r.get('extracted_frames', [])),
                            'error': r.get('error') if r.get('status') == 'error' else None
                        }
                        for r in all_results
                    ]
                }
                
                # 디버깅용 로그 추가
                current_app.logger.info(f"Sending processing_complete event with {len(all_frames)} frames")
                current_app.logger.info(f"final_data keys: {list(final_data.keys())}")
                
                yield f"data: {json.dumps({'type': 'processing_complete', 'data': final_data})}\n\n"
                
                # 실제 추출 결과를 기존 UI와 호환되는 형식으로 전송
                if all_frames:
                    # LLM-IE 호환 문서 구조 생성
                    llm_ie_document = {
                        'doc_id': 'gemini_batch_extraction',
                        'text': input_text,
                        'frames': all_frames,
                        'relations': []
                    }
                    
                    current_app.logger.info(f"Sending result event with {len(all_frames)} frames")
                    yield f"data: {json.dumps({'type': 'result', 'frames': all_frames, 'document': llm_ie_document})}\n\n"
                else:
                    current_app.logger.warning("No frames to send in result event")
                
            except Exception as e:
                current_app.logger.error(f"Error during Gemini batch processing: {e}\n{traceback.format_exc()}")
                yield f"data: {json.dumps({'type': 'batch_error', 'message': f'Gemini batch processing failed: {str(e)}'})}\n\n"
            finally:
                yield "event: end\ndata: {}\n\n"
        
        return Response(stream_with_context(generate_gemini_batch_stream()), mimetype='text/event-stream')
        
    except Exception as e:
        current_app.logger.error(f"Failed to setup Gemini batch processing: {e}")
        return jsonify({"error": f"Gemini batch processing setup failed: {str(e)}"}), 500


@main_bp.route('/api/frame-extraction/gemini-single', methods=['POST'])
def api_frame_extraction_gemini_single():
    """
    Gemini 직접 호출을 사용한 단일 추출 엔드포인트
    """
    data = request.json
    input_text = data.get('inputText', '')
    prompt_template = data.get('promptTemplate', '')
    gemini_config = data.get('geminiConfig', {})
    
    api_key = gemini_config.get('apiKey', '')
    gemini_model = gemini_config.get('model', 'gemini-2.0-flash')
    temperature = float(gemini_config.get('temperature', 0.2))
    max_tokens = int(gemini_config.get('maxTokens', 4096))
    
    if not input_text:
        return jsonify({"error": "Input text is required"}), 400
    if not prompt_template:
        return jsonify({"error": "Prompt template is required"}), 400
    if not api_key:
        return jsonify({"error": "Gemini API key is required"}), 400
    
    try:
        from .gemini_direct_engine import GeminiDirectConfig, GeminiDirectEngine
        
        config = GeminiDirectConfig(
            api_key=api_key,
            model=gemini_model,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        engine = GeminiDirectEngine(config)
        
        def generate_gemini_single_stream():
            try:
                yield f"data: {json.dumps({'type': 'info', 'data': 'Starting Gemini Direct extraction...'})}\n\n"
                
                # 프롬프트 템플릿에서 {{text}} 대체
                full_prompt = prompt_template.replace('{{text}}', input_text)
                
                messages = [
                    {'role': 'user', 'content': full_prompt}
                ]
                
                yield f"data: {json.dumps({'type': 'info', 'data': 'Sending request to Gemini...'})}\n\n"
                
                # Gemini API 호출
                response_generator = engine.chat_completion(messages, stream=False)
                
                for response in response_generator:
                    if response.get('type') == 'content' and response.get('done'):
                        raw_text = response.get('text', '').strip()
                        yield f"data: {json.dumps({'type': 'info', 'data': 'Processing Gemini response...'})}\n\n"
                        
                        # JSON 응답 파싱 (소스 텍스트 전달)
                        from .gemini_direct_engine import GeminiDirectBatchProcessor
                        processor = GeminiDirectBatchProcessor([api_key], {'temperature': temperature})
                        extracted_frames = processor._parse_extraction_response(raw_text, input_text)
                        
                        yield f"data: {json.dumps({'type': 'info', 'data': f'Extracted {len(extracted_frames)} frames'})}\n\n"
                        
                        # 결과 전송
                        if extracted_frames:
                            # LLM-IE 호환 문서 구조 생성
                            llm_ie_document = {
                                'doc_id': 'gemini_single_extraction',
                                'text': input_text,
                                'frames': extracted_frames,
                                'relations': []
                            }
                            yield f"data: {json.dumps({'type': 'result', 'frames': extracted_frames, 'document': llm_ie_document})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'warning', 'data': 'No frames extracted'})}\n\n"
                            yield f"data: {json.dumps({'type': 'debug', 'data': f'Raw response: {raw_text}'})}\n\n"
                        
                        break
                        
                    elif response.get('type') == 'error':
                        yield f"data: {json.dumps({'type': 'error', 'message': response.get('error', 'Unknown error')})}\n\n"
                        break
                
            except Exception as e:
                current_app.logger.error(f"Error during Gemini single extraction: {e}\n{traceback.format_exc()}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Gemini extraction failed: {str(e)}'})}\n\n"
            finally:
                yield "event: end\ndata: {}\n\n"
        
        return Response(stream_with_context(generate_gemini_single_stream()), mimetype='text/event-stream')
        
    except Exception as e:
        current_app.logger.error(f"Failed to setup Gemini single extraction: {e}")
        return jsonify({"error": f"Gemini extraction setup failed: {str(e)}"}), 500


# ===============================
# Knowledge Graph API Endpoints
# ===============================

@main_bp.route('/api/knowledge-graph/generate', methods=['POST'])
def api_knowledge_graph_generate():
    """
    지식그래프 생성 API 엔드포인트
    LLM-IE 추출 결과로부터 RDF 지식그래프를 생성
    """
    try:
        data = request.json
        extraction_data = data.get('extraction_data', {})
        settings = data.get('settings', {})
        
        if not extraction_data:
            return jsonify({"error": "extraction_data is required"}), 400
        
        current_app.logger.info("Starting knowledge graph generation")
        current_app.logger.info(f"Extraction data contains {len(extraction_data.get('frames', []))} frames")
        
        # LLM 엔진 설정
        llm_engine = None
        if settings.get('enable_llm_inference', False):
            try:
                llm_api_type = settings.get('llm_api_type', 'gemini_direct')
                
                if llm_api_type == 'gemini_direct':
                    # Gemini Direct API 사용
                    api_key = settings.get('gemini_api_key')
                    if not api_key:
                        return jsonify({"error": "Gemini API key is required for LLM inference"}), 400
                    
                    from .gemini_direct_engine import GeminiDirectConfig, GeminiDirectEngine
                    config = GeminiDirectConfig(
                        api_key=api_key,
                        model='gemini-2.0-flash',
                        temperature=0.2,
                        max_output_tokens=4096
                    )
                    llm_engine = GeminiDirectEngine(config)
                else:
                    # 기존 LLM 엔진 사용
                    llm_config = {
                        'api_type': llm_api_type,
                        'temperature': 0.2,
                        'max_new_tokens': 4096
                    }
                    llm_engine = create_llm_engine_from_config(llm_config)
                
                current_app.logger.info(f"LLM engine configured: {llm_api_type}")
            except Exception as e:
                current_app.logger.warning(f"Failed to configure LLM engine: {e}")
                # LLM 없이 진행
                llm_engine = None
        
        # 지식그래프 생성기 초기화
        kg_generator = KnowledgeGraphGenerator(llm_engine=llm_engine)
        
        # Multi-agent 매개변수 추출
        user_context = settings.get('user_context', '')
        domain = settings.get('domain', 'general')
        additional_guidelines = settings.get('additional_guidelines', '')
        max_relations = settings.get('max_relations', 20)
        use_iterative_inference = settings.get('use_iterative_inference', True)
        enable_entity_resolution = settings.get('enable_entity_resolution', True)
        
        # 관계 추출 컨텍스트 설정 처리
        relation_context = settings.get('relation_context', {})
        relation_domain = relation_context.get('domain', 'general')
        relation_template = relation_context.get('template', 'detailed')
        context_prompt = relation_context.get('context', '')
        relation_goal = relation_context.get('goal', '')
        relation_constraints = relation_context.get('constraints', '')
        
        # 컨텍스트가 제공된 경우 user_context와 additional_guidelines에 통합
        if context_prompt or relation_goal or relation_constraints:
            context_parts = []
            if context_prompt:
                context_parts.append(f"문서 컨텍스트: {context_prompt}")
            if relation_goal:
                context_parts.append(f"관계 추출 목표: {relation_goal}")
            if relation_constraints:
                context_parts.append(f"추출 제약사항: {relation_constraints}")
            
            enhanced_context = " | ".join(context_parts)
            if user_context:
                user_context = f"{user_context} | {enhanced_context}"
            else:
                user_context = enhanced_context
        
        # 도메인과 템플릿 정보를 additional_guidelines에 추가
        if relation_domain != 'general' or relation_template != 'detailed':
            template_guidance = {
                'detailed': '관계의 세부사항과 맥락을 포함하여 상세하게 추출',
                'concise': '핵심적인 관계만을 간결하게 식별',
                'exploratory': '다양한 관계 유형을 탐색하여 숨겨진 연결점을 발견'
            }
            
            domain_guidance = f"도메인: {relation_domain}"
            template_desc = template_guidance.get(relation_template, '표준 추출 방식')
            
            guidance_text = f"{domain_guidance}, 추출 방식: {template_desc}"
            if additional_guidelines:
                additional_guidelines = f"{additional_guidelines} | {guidance_text}"
            else:
                additional_guidelines = guidance_text
        
        # 지식그래프 생성 (Multi-agent 시스템 사용)
        result = kg_generator.generate_knowledge_graph(
            extraction_data=extraction_data,
            user_context=user_context,
            domain=domain,
            additional_guidelines=additional_guidelines,
            max_relations=max_relations,
            use_iterative_inference=use_iterative_inference,
            enable_entity_resolution=enable_entity_resolution
        )
        
        if result['success']:
            current_app.logger.info(f"Knowledge graph generated successfully with {result['total_triples']} triples")
        else:
            current_app.logger.error(f"Knowledge graph generation failed: {result.get('error')}")
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Error in knowledge graph generation API: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Knowledge graph generation failed: {str(e)}",
            "timestamp": time.time()
        }), 500


@main_bp.route('/api/knowledge-graph/validate', methods=['POST'])
def api_knowledge_graph_validate():
    """
    지식그래프 검증 API 엔드포인트
    생성된 지식그래프의 품질과 일관성을 검증
    """
    try:
        data = request.json
        kg_data = data.get('kg_data', {})
        settings = data.get('settings', {})
        
        if not kg_data:
            return jsonify({"error": "kg_data is required"}), 400
        
        current_app.logger.info("Starting knowledge graph validation")
        
        # LLM 엔진 설정 (검증용)
        llm_engine = None
        if settings.get('enable_llm_validation', False):
            try:
                llm_api_type = settings.get('llm_api_type', 'gemini_direct')
                
                if llm_api_type == 'gemini_direct':
                    # Gemini Direct API 사용
                    api_key = settings.get('gemini_api_key')
                    if not api_key:
                        return jsonify({"error": "Gemini API key is required for LLM validation"}), 400
                    
                    from .gemini_direct_engine import GeminiDirectConfig, GeminiDirectEngine
                    config = GeminiDirectConfig(
                        api_key=api_key,
                        model='gemini-2.0-flash',
                        temperature=0.1,  # 검증에는 낮은 temperature 사용
                        max_output_tokens=2048
                    )
                    llm_engine = GeminiDirectEngine(config)
                else:
                    # 기존 LLM 엔진 사용
                    llm_config = {
                        'api_type': llm_api_type,
                        'temperature': 0.1,
                        'max_new_tokens': 2048
                    }
                    llm_engine = create_llm_engine_from_config(llm_config)
                
                current_app.logger.info(f"LLM engine configured for validation: {llm_api_type}")
            except Exception as e:
                current_app.logger.warning(f"Failed to configure LLM engine for validation: {e}")
                # LLM 없이 진행
                llm_engine = None
        
        # 지식그래프 검증기 초기화
        kg_validator = KnowledgeGraphValidator(llm_engine=llm_engine)
        
        # 자동 오류 수정 설정
        auto_fix_errors = settings.get('auto_fix_errors', True)
        
        # 지식그래프 검증 (자동 수정 포함)
        result = kg_validator.validate_knowledge_graph(kg_data, auto_fix_errors=auto_fix_errors)
        
        if result['success']:
            current_app.logger.info(f"Knowledge graph validation completed with score: {result['overall_score']}")
        else:
            current_app.logger.error(f"Knowledge graph validation failed: {result.get('error')}")
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Error in knowledge graph validation API: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Knowledge graph validation failed: {str(e)}",
            "timestamp": time.time()
        }), 500


@main_bp.route('/api/knowledge-graph/load-sample', methods=['GET'])
def api_knowledge_graph_load_sample():
    """
    샘플 지식그래프 데이터 로드 (테스트 및 데모 목적)
    """
    try:
        # 샘플 LLM-IE 추출 결과 생성
        sample_data = {
            "doc_id": "sample_kg_document",
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
                },
                {
                    "frame_id": "5",
                    "start": 71,
                    "end": 75,
                    "entity_text": "프롬프트",
                    "attr": {"entity_type": "TECHNOLOGY_PARAMETER"}
                },
                {
                    "frame_id": "6",
                    "start": 77,
                    "end": 92,
                    "entity_text": "온도(temperature)",
                    "attr": {"entity_type": "TECHNOLOGY_PARAMETER"}
                },
                {
                    "frame_id": "7",
                    "start": 111,
                    "end": 127,
                    "entity_text": "IPIP-NEO-120 설문지",
                    "attr": {"entity_type": "SURVEY"}
                },
                {
                    "frame_id": "8",
                    "start": 136,
                    "end": 138,
                    "entity_text": "연구",
                    "attr": {"entity_type": "RESEARCH"}
                }
            ],
            "relations": [
                {
                    "subject": "1",
                    "relation_type": "IS_TYPE_OF",
                    "object": "4",
                    "confidence": 0.95
                },
                {
                    "subject": "2",
                    "relation_type": "IS_TYPE_OF",
                    "object": "4",
                    "confidence": 0.95
                },
                {
                    "subject": "3",
                    "relation_type": "IS_TYPE_OF",
                    "object": "4",
                    "confidence": 0.95
                },
                {
                    "subject": "8",
                    "relation_type": "STUDIES",
                    "object": "4",
                    "confidence": 0.9
                },
                {
                    "subject": "8",
                    "relation_type": "USES",
                    "object": "7",
                    "confidence": 0.85
                }
            ]
        }
        
        return jsonify({
            "success": True,
            "sample_data": sample_data,
            "description": "LLM 성격 분석 연구 관련 샘플 데이터"
        })
        
    except Exception as e:
        current_app.logger.error(f"Error loading sample knowledge graph data: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@main_bp.route('/api/relation-inference/iterative', methods=['POST'])
def api_relation_inference_iterative():
    """
    반복적 관계 추론 API 엔드포인트
    Multi-agent 시스템을 사용하여 고품질 관계를 추론
    """
    try:
        data = request.json
        frames = data.get('frames', [])
        text = data.get('text', '')
        settings = data.get('settings', {})
        
        if not frames or not text:
            return jsonify({"error": "frames and text are required"}), 400
        
        current_app.logger.info(f"Starting iterative relation inference for {len(frames)} entities")
        
        # LLM 엔진 설정
        try:
            llm_api_type = settings.get('llm_api_type', 'gemini_direct')
            
            if llm_api_type == 'gemini_direct':
                api_key = settings.get('gemini_api_key')
                if not api_key:
                    return jsonify({"error": "Gemini API key is required for relation inference"}), 400
                
                from .gemini_direct_engine import GeminiDirectConfig, GeminiDirectEngine
                config = GeminiDirectConfig(
                    api_key=api_key,
                    model='gemini-2.0-flash',
                    temperature=0.3,
                    max_output_tokens=8192
                )
                llm_engine = GeminiDirectEngine(config)
            else:
                llm_config = {
                    'api_type': llm_api_type,
                    'temperature': 0.3,
                    'max_new_tokens': 8192
                }
                llm_engine = create_llm_engine_from_config(llm_config)
            
            current_app.logger.info(f"LLM engine configured for relation inference: {llm_api_type}")
        except Exception as e:
            current_app.logger.error(f"Failed to configure LLM engine: {e}")
            return jsonify({"error": f"LLM configuration failed: {str(e)}"}), 400
        
        # IterativeRelationInferenceAgent 사용
        try:
            from .relation_agents import IterativeRelationInferenceAgent
            
            relation_agent = IterativeRelationInferenceAgent(llm_engine)
            
            # 관계 추론 실행
            result = relation_agent.infer_relations(
                frames=frames,
                text=text,
                user_context=settings.get('user_context', ''),
                domain=settings.get('domain', 'general'),
                additional_guidelines=settings.get('additional_guidelines', ''),
                max_relations=settings.get('max_relations', 20),
                target_score=settings.get('target_score', 7.5),
                max_iterations=settings.get('max_iterations', 3)
            )
            
            if result.get('success'):
                current_app.logger.info(f"Iterative relation inference completed: {len(result.get('relations', []))} relations")
            else:
                current_app.logger.error(f"Iterative relation inference failed: {result.get('error')}")
            
            return jsonify(result)
            
        except ImportError:
            return jsonify({"error": "IterativeRelationInferenceAgent not available"}), 500
        except Exception as e:
            current_app.logger.error(f"Error in iterative relation inference: {e}", exc_info=True)
            return jsonify({"error": f"Relation inference failed: {str(e)}"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error in iterative relation inference API: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"API request failed: {str(e)}"
        }), 500


@main_bp.route('/api/relations/edit', methods=['POST'])
def api_relations_edit():
    """
    관계 편집 API 엔드포인트
    사용자가 관계를 추가, 수정, 삭제할 수 있는 인터페이스
    """
    try:
        data = request.json
        action = data.get('action')  # 'create', 'update', 'delete', 'bulk_approve', 'bulk_reject'
        relation_data = data.get('relation_data', {})
        relations_list = data.get('relations_list', [])
        
        current_app.logger.info(f"Processing relation edit action: {action}")
        
        if action == 'create':
            # 새 관계 생성
            new_relation = {
                'id': f"rel_{int(time.time())}",
                'subject': relation_data.get('subject'),
                'relation_type': relation_data.get('relation_type'),
                'object': relation_data.get('object'),
                'confidence': relation_data.get('confidence', 0.5),
                'explanation': relation_data.get('explanation', ''),
                'status': 'approved',  # 사용자가 직접 생성한 관계는 승인됨
                'created_by': 'user'
            }
            
            return jsonify({
                'success': True,
                'action': 'create',
                'relation': new_relation
            })
            
        elif action == 'update':
            # 관계 업데이트
            updated_relation = relation_data.copy()
            updated_relation['modified_at'] = time.time()
            
            return jsonify({
                'success': True,
                'action': 'update',
                'relation': updated_relation
            })
            
        elif action == 'delete':
            # 관계 삭제
            relation_id = relation_data.get('id')
            if not relation_id:
                return jsonify({"error": "Relation ID is required for deletion"}), 400
            
            return jsonify({
                'success': True,
                'action': 'delete',
                'relation_id': relation_id
            })
            
        elif action == 'bulk_approve':
            # 대량 승인
            approved_count = 0
            for relation in relations_list:
                if relation.get('status') != 'approved':
                    relation['status'] = 'approved'
                    relation['modified_at'] = time.time()
                    approved_count += 1
            
            return jsonify({
                'success': True,
                'action': 'bulk_approve',
                'approved_count': approved_count,
                'relations': relations_list
            })
            
        elif action == 'bulk_reject':
            # 낮은 점수 관계 대량 거절
            threshold = relation_data.get('score_threshold', 6.0)
            rejected_count = 0
            
            for relation in relations_list:
                if relation.get('confidence', 0) * 10 < threshold and relation.get('status') != 'rejected':
                    relation['status'] = 'rejected'
                    relation['modified_at'] = time.time()
                    rejected_count += 1
            
            return jsonify({
                'success': True,
                'action': 'bulk_reject',
                'threshold': threshold,
                'rejected_count': rejected_count,
                'relations': relations_list
            })
            
        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400
        
    except Exception as e:
        current_app.logger.error(f"Error in relations edit API: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Relation edit failed: {str(e)}"
        }), 500


@main_bp.route('/api/relations/validate-batch', methods=['POST'])
def api_relations_validate_batch():
    """
    관계 일괄 검증 API 엔드포인트
    LLM을 사용하여 다수의 관계를 한번에 검증
    """
    try:
        data = request.json
        relations = data.get('relations', [])
        settings = data.get('settings', {})
        
        if not relations:
            return jsonify({"error": "relations list is required"}), 400
        
        current_app.logger.info(f"Starting batch validation for {len(relations)} relations")
        
        # LLM 엔진 설정
        try:
            llm_api_type = settings.get('llm_api_type', 'gemini_direct')
            
            if llm_api_type == 'gemini_direct':
                api_key = settings.get('gemini_api_key')
                if not api_key:
                    return jsonify({"error": "Gemini API key is required for validation"}), 400
                
                from .gemini_direct_engine import GeminiDirectConfig, GeminiDirectEngine
                config = GeminiDirectConfig(
                    api_key=api_key,
                    model='gemini-2.0-flash',
                    temperature=0.1,
                    max_output_tokens=4096
                )
                llm_engine = GeminiDirectEngine(config)
            else:
                llm_config = {
                    'api_type': llm_api_type,
                    'temperature': 0.1,
                    'max_new_tokens': 4096
                }
                llm_engine = create_llm_engine_from_config(llm_config)
                
        except Exception as e:
            return jsonify({"error": f"LLM configuration failed: {str(e)}"}), 400
        
        # RelationEvaluatorAgent 사용
        try:
            from .relation_agents import RelationEvaluatorAgent
            
            evaluator = RelationEvaluatorAgent(llm_engine)
            
            # 관계 일괄 평가
            validated_relations = []
            for relation in relations:
                try:
                    evaluation_result = evaluator.evaluate_relation(
                        subject_entity=relation.get('subject_text', ''),
                        object_entity=relation.get('object_text', ''),
                        relation_type=relation.get('relation_type'),
                        context=settings.get('text', ''),
                        explanation=relation.get('explanation', '')
                    )
                    
                    if evaluation_result.get('success'):
                        relation['validation_score'] = evaluation_result.get('overall_score', 0)
                        relation['validation_feedback'] = evaluation_result.get('feedback', '')
                        relation['validation_suggestions'] = evaluation_result.get('suggestions', [])
                        relation['validated_at'] = time.time()
                    else:
                        relation['validation_error'] = evaluation_result.get('error')
                    
                    validated_relations.append(relation)
                    
                except Exception as rel_error:
                    current_app.logger.error(f"Error validating relation {relation.get('id')}: {rel_error}")
                    relation['validation_error'] = str(rel_error)
                    validated_relations.append(relation)
            
            return jsonify({
                'success': True,
                'validated_relations': validated_relations,
                'total_processed': len(validated_relations)
            })
            
        except ImportError:
            return jsonify({"error": "RelationEvaluatorAgent not available"}), 500
        except Exception as e:
            current_app.logger.error(f"Error in batch validation: {e}", exc_info=True)
            return jsonify({"error": f"Batch validation failed: {str(e)}"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error in relations validate batch API: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"API request failed: {str(e)}"
        }), 500


@main_bp.route('/api/knowledge-graph/manual-relation', methods=['POST'])
def api_add_manual_relation():
    """
    수동 관계 추가 API 엔드포인트
    사용자가 직접 생성한 관계를 지식그래프에 추가
    """
    try:
        data = request.json
        relation_data = data.get('relation', {})
        
        if not relation_data:
            return jsonify({"error": "relation data is required"}), 400
            
        # 필수 필드 검증
        required_fields = ['from', 'to', 'label']
        for field in required_fields:
            if field not in relation_data:
                return jsonify({"error": f"'{field}' field is required"}), 400
        
        current_app.logger.info(f"Adding manual relation: {relation_data['from']} -> {relation_data['to']} ({relation_data['label']})")
        
        # 관계 데이터 정규화
        manual_relation = {
            'id': relation_data.get('id', f"manual_rel_{int(time.time())}"),
            'from': relation_data['from'],
            'to': relation_data['to'],
            'label': relation_data['label'],
            'description': relation_data.get('description', ''),
            'confidence': float(relation_data.get('confidence', 0.9)),
            'direction': relation_data.get('direction', 'directed'),
            'manual': True,
            'created_at': time.time()
        }
        
        # 관계가 유효한지 검증 (간단한 검증)
        if manual_relation['from'] == manual_relation['to']:
            return jsonify({"error": "Self-referencing relations are not allowed"}), 400
            
        if not manual_relation['label'].strip():
            return jsonify({"error": "Relation label cannot be empty"}), 400
            
        current_app.logger.info("Manual relation added successfully")
        
        return jsonify({
            "success": True,
            "relation": manual_relation,
            "message": "Manual relation added successfully"
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in manual relation API: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Failed to add manual relation: {str(e)}"
        }), 500


@main_bp.route('/api/knowledge-graph/export-with-manual-relations', methods=['POST'])
def api_export_knowledge_graph_with_manual_relations():
    """
    수동 관계가 포함된 지식그래프 내보내기 API
    """
    try:
        data = request.json
        kg_data = data.get('kg_data', {})
        manual_relations = data.get('manual_relations', [])
        export_format = data.get('format', 'json')
        
        if not kg_data:
            return jsonify({"error": "kg_data is required"}), 400
        
        current_app.logger.info(f"Exporting knowledge graph with {len(manual_relations)} manual relations")
        
        # 원본 지식그래프 데이터에 수동 관계 추가
        enhanced_kg_data = kg_data.copy()
        
        # 기존 관계에 수동 관계 추가
        existing_relations = enhanced_kg_data.get('relations', [])
        all_relations = existing_relations + manual_relations
        enhanced_kg_data['relations'] = all_relations
        
        # 통계 업데이트
        if 'statistics' in enhanced_kg_data:
            enhanced_kg_data['statistics']['total_relations'] = len(all_relations)
            enhanced_kg_data['statistics']['manual_relations'] = len(manual_relations)
        
        # 메타데이터 추가
        enhanced_kg_data['export_info'] = {
            'exported_at': time.time(),
            'includes_manual_relations': len(manual_relations) > 0,
            'total_relations': len(all_relations),
            'manual_relations_count': len(manual_relations)
        }
        
        if export_format == 'json':
            return jsonify({
                "success": True,
                "data": enhanced_kg_data,
                "format": "json"
            })
        else:
            return jsonify({"error": f"Export format '{export_format}' is not supported"}), 400
        
    except Exception as e:
        current_app.logger.error(f"Error in export knowledge graph API: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Export failed: {str(e)}"
        }), 500

