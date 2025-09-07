# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-IE is a comprehensive toolkit for LLM-based information extraction, providing building blocks for named entity recognition, attribute extraction, and relation extraction pipelines. The project consists of three main components:

1. **Core Python Package** (`package/llm-ie/`): The main library with extractors, chunkers, engines, and data types
2. **Web Application** (`web_app/`): Flask-based UI for drag-and-drop access to LLM-IE functionality  
3. **Documentation** (`docs/`): MkDocs-based documentation site

## Architecture Overview

### Core Components

- **Engines** (`engines.py`): Abstract inference engine interfaces supporting multiple LLM providers (OpenAI, Ollama, vLLM, HuggingFace, etc.)
- **Extractors** (`extractors.py`): Frame and relation extraction classes with different prompting strategies:
  - `DirectFrameExtractor`: Single-pass extraction
  - `ReviewFrameExtractor`: Multi-pass with review step
  - `AttributeExtractor`: Entity attribute extraction
  - `BinaryRelationExtractor`/`MultiClassRelationExtractor`: Relationship extraction
- **Chunkers** (`chunkers.py`): Text segmentation strategies:
  - `UnitChunker`: Defines processing units (sentence, document, line)
  - `ContextChunker`: Provides context around units (sliding window, whole document)
- **Data Types** (`data_types.py`): Core data structures for frames and documents
- **Prompt Editor** (`prompt_editor.py`): Interactive LLM agent for prompt engineering

### Processing Pipeline

1. **Text Chunking**: Documents are split into units (sentences/paragraphs) with optional context
2. **LLM Inference**: Each unit is processed through the configured extraction pipeline  
3. **Frame Collection**: Extracted entities/relations are collected as `LLMInformationExtractionFrame` objects
4. **Document Assembly**: Frames are aggregated into `LLMInformationExtractionDocument` for management and visualization

## Development Commands

### Python Package Development
```bash
# Navigate to package directory
cd package/llm-ie/

# Install dependencies (using Poetry)
poetry install

# Build package
poetry build

# Install in development mode
pip install -e .
```

### Web Application Development
```bash
# Navigate to web app directory
cd web_app/

# Install dependencies
pip install -r requirements.txt

# Run development server
python run.py

# Run with Gunicorn (production)
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

### Documentation
```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### Docker (Web App)
```bash
# Build image
docker build -t llm-ie-web-app web_app/

# Run container
docker run -p 5000:5000 llm-ie-web-app
```

## Key Configuration Files

- `package/llm-ie/pyproject.toml`: Python package configuration with Poetry
- `web_app/requirements.txt`: Web application dependencies
- `mkdocs.yml`: Documentation site configuration
- `package/llm-ie/pipelines/sample_config.yaml`: Sample pipeline configuration

## LLM Engine Configuration

The project supports multiple LLM providers through a unified interface. Each engine requires specific configuration:

- **OpenAI**: Requires `OPENAI_API_KEY` environment variable
- **Azure OpenAI**: Requires endpoint and API version configuration  
- **Ollama**: Connects to local Ollama instance
- **vLLM**: Uses OpenAI-compatible server interface
- **HuggingFace**: Direct model loading via hub

All engines support both `BasicLLMConfig` and `ReasoningLLMConfig` for reasoning models (o3, Qwen3).

## Pipeline Processing

For batch processing, use the pipeline scripts:
- `package/llm-ie/pipelines/sequential_frame_extraction.py`: Single-threaded processing
- `package/llm-ie/pipelines/multithread_frame_extraction.py`: Multi-threaded processing

Configure pipeline parameters in YAML files following the sample format.

## Asset Management

- Default prompts: `package/llm-ie/src/llm_ie/asset/default_prompts/`
- Prompt guides: `package/llm-ie/src/llm_ie/asset/prompt_guide/`
- PromptEditor prompts: `package/llm-ie/src/llm_ie/asset/PromptEditor_prompts/`

## Web Application Structure

- `app/routes.py`: Flask route definitions
- `app/extractors.py`: Web-specific extraction logic  
- `app/templates/`: Jinja2 templates for UI components
- `app/static/`: CSS, JavaScript, and image assets