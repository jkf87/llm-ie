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

