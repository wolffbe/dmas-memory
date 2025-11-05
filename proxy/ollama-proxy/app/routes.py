import json
import logging
import time
from typing import Any, Dict
import requests
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from metrics import track_metrics, ollama_active_models, get_stats
from utils import log_request

logger = logging.getLogger(__name__)


def register_routes(app: FastAPI) -> None:
    
    @app.post('/v1/chat/completions')
    async def openai_chat_completions(request: Request):
        request_data = await request.json()
        model = request_data.get('model', 'unknown')
        messages = request_data.get('messages', [])
        stream = request_data.get('stream', False)
        tools = request_data.get('tools', [])
        tool_choice = request_data.get('tool_choice')
        
        logger.info(f"[OpenAI] {model}: {json.dumps(messages)}")
        
        ollama_host = request.app.state.config['ollama_host']
        log_dir = request.app.state.config['log_dir']
        
        ollama_request = {
            'model': model,
            'messages': messages,
            'stream': stream
        }
        
        if tools:
            ollama_request['tools'] = tools
        if tool_choice:
            ollama_request['tool_choice'] = tool_choice
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f'{ollama_host}/api/chat',
                json=ollama_request,
                stream=stream,
                timeout=300
            )
            
            if not stream:
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    ollama_response = response.json()
                    
                    openai_response = {
                        'id': f'chatcmpl-{int(time.time())}',
                        'object': 'chat.completion',
                        'created': int(time.time()),
                        'model': model,
                        'choices': [{
                            'index': 0,
                            'message': ollama_response.get('message', {}),
                            'finish_reason': 'stop'
                        }],
                        'usage': {
                            'prompt_tokens': ollama_response.get('prompt_eval_count', 0),
                            'completion_tokens': ollama_response.get('eval_count', 0),
                            'total_tokens': ollama_response.get('prompt_eval_count', 0) + ollama_response.get('eval_count', 0)
                        }
                    }
                    
                    track_metrics(model, 'chat', ollama_response, duration, 'success')
                    log_request('openai_chat', request_data, openai_response, duration, 'success', log_dir)
                    
                    return JSONResponse(content=openai_response)
                else:
                    track_metrics(model, 'chat', {}, duration, 'error')
                    log_request('openai_chat', request_data, {'error': response.text}, duration, 'error', log_dir)
                    raise HTTPException(status_code=response.status_code, detail=response.text)
            
            else:
                async def stream_openai_format():
                    try:
                        for line in response.iter_lines():
                            if line:
                                chunk = json.loads(line)
                                
                                openai_chunk = {
                                    'id': f'chatcmpl-{int(time.time())}',
                                    'object': 'chat.completion.chunk',
                                    'created': int(time.time()),
                                    'model': model,
                                    'choices': [{
                                        'index': 0,
                                        'delta': chunk.get('message', {}),
                                        'finish_reason': None
                                    }]
                                }
                                
                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                        
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        logger.error(f"Error in streaming: {e}")
                
                return StreamingResponse(
                    stream_openai_format(),
                    media_type='text/event-stream'
                )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[OpenAI] Error: {e}")
            track_metrics(model, 'chat', {}, duration, 'error')
            log_request('openai_chat', request_data, {'error': str(e)}, duration, 'error', log_dir)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get('/v1/models')
    async def openai_list_models(request: Request):
        ollama_host = request.app.state.config['ollama_host']
        
        try:
            response = requests.get(f'{ollama_host}/api/tags', timeout=10)
            
            if response.status_code == 200:
                ollama_data = response.json()
                ollama_models = ollama_data.get('models', [])
                
                # Convert to OpenAI format
                openai_models = {
                    'object': 'list',
                    'data': [
                        {
                            'id': model['name'],
                            'object': 'model',
                            'created': int(time.time()),
                            'owned_by': 'ollama'
                        }
                        for model in ollama_models
                    ]
                }
                
                return JSONResponse(content=openai_models)
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post('/api/generate')
    async def generate(request: Request):
        request_data = await request.json()
        logger.info(f"Received /api/generate request: {request_data}")
        model = request_data.get('model', 'unknown')
        stream = request_data.get('stream', False)
        ollama_host = request.app.state.config['ollama_host']
        log_dir = request.app.state.config['log_dir']
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f'{ollama_host}/api/generate',
                json=request_data,
                stream=stream,
                timeout=300
            )
            
            if not stream:
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    response_data = response.json()
                    track_metrics(model, 'generate', response_data, duration, 'success')
                    log_request('generate', request_data, response_data, duration, 'success', log_dir)
                else:
                    track_metrics(model, 'generate', {}, duration, 'error')
                    log_request('generate', request_data, {'error': response.text}, duration, 'error', log_dir)
                
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    media_type='application/json'
                )
            
            else:
                async def generate_with_tracking():
                    accumulated_response = {
                        'prompt_eval_count': 0,
                        'eval_count': 0,
                        'response': '',
                        'total_duration': 0,
                        'load_duration': 0,
                        'prompt_eval_duration': 0,
                        'eval_duration': 0,
                    }
                    
                    try:
                        for line in response.iter_lines():
                            if line:
                                chunk = json.loads(line)
                                
                                if 'prompt_eval_count' in chunk:
                                    accumulated_response['prompt_eval_count'] = chunk['prompt_eval_count']
                                if 'eval_count' in chunk:
                                    accumulated_response['eval_count'] = chunk['eval_count']
                                if 'response' in chunk:
                                    accumulated_response['response'] += chunk['response']
                                if 'total_duration' in chunk:
                                    accumulated_response['total_duration'] = chunk['total_duration']
                                if 'load_duration' in chunk:
                                    accumulated_response['load_duration'] = chunk['load_duration']
                                if 'prompt_eval_duration' in chunk:
                                    accumulated_response['prompt_eval_duration'] = chunk['prompt_eval_duration']
                                if 'eval_duration' in chunk:
                                    accumulated_response['eval_duration'] = chunk['eval_duration']
                                
                                yield line + b'\n'
                        
                        duration = time.time() - start_time
                        track_metrics(model, 'generate', accumulated_response, duration, 'success')
                        log_request('generate', request_data, accumulated_response, duration, 'success', log_dir)
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        logger.error(f"Error in streaming: {e}")
                        track_metrics(model, 'generate', {}, duration, 'error')
                        log_request('generate', request_data, {'error': str(e)}, duration, 'error', log_dir)
                
                return StreamingResponse(
                    generate_with_tracking(),
                    media_type='application/x-ndjson'
                )
            
        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            logger.error(f"Ollama request timeout after {duration:.2f}s")
            track_metrics(model, 'generate', {}, duration, 'timeout')
            log_request('generate', request_data, {'error': 'timeout'}, duration, 'timeout', log_dir)
            raise HTTPException(status_code=504, detail='Request timeout')
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error proxying request: {e}")
            track_metrics(model, 'generate', {}, duration, 'error')
            log_request('generate', request_data, {'error': str(e)}, duration, 'error', log_dir)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post('/api/chat')
    async def chat(request: Request):
        request_data = await request.json()
        model = request_data.get('model', 'unknown')
        stream = request_data.get('stream', False)
        ollama_host = request.app.state.config['ollama_host']
        log_dir = request.app.state.config['log_dir']
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f'{ollama_host}/api/chat',
                json=request_data,
                stream=stream,
                timeout=300
            )
            
            if not stream:
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    response_data = response.json()
                    track_metrics(model, 'chat', response_data, duration, 'success')
                    log_request('chat', request_data, response_data, duration, 'success', log_dir)
                else:
                    track_metrics(model, 'chat', {}, duration, 'error')
                    log_request('chat', request_data, {'error': response.text}, duration, 'error', log_dir)
                
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    media_type='application/json'
                )
            
            else:
                async def chat_with_tracking():
                    accumulated_response = {
                        'prompt_eval_count': 0,
                        'eval_count': 0,
                        'message': {'role': '', 'content': ''},
                        'total_duration': 0,
                        'load_duration': 0,
                        'eval_duration': 0,
                    }
                    
                    try:
                        for line in response.iter_lines():
                            if line:
                                chunk = json.loads(line)
                                
                                if 'prompt_eval_count' in chunk:
                                    accumulated_response['prompt_eval_count'] = chunk['prompt_eval_count']
                                if 'eval_count' in chunk:
                                    accumulated_response['eval_count'] = chunk['eval_count']
                                if 'message' in chunk:
                                    accumulated_response['message']['content'] += chunk['message'].get('content', '')
                                    accumulated_response['message']['role'] = chunk['message'].get('role', '')
                                if 'total_duration' in chunk:
                                    accumulated_response['total_duration'] = chunk['total_duration']
                                if 'load_duration' in chunk:
                                    accumulated_response['load_duration'] = chunk['load_duration']
                                if 'eval_duration' in chunk:
                                    accumulated_response['eval_duration'] = chunk['eval_duration']
                                
                                yield line + b'\n'
                        
                        duration = time.time() - start_time
                        track_metrics(model, 'chat', accumulated_response, duration, 'success')
                        log_request('chat', request_data, accumulated_response, duration, 'success', log_dir)
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        logger.error(f"Error in streaming: {e}")
                        track_metrics(model, 'chat', {}, duration, 'error')
                        log_request('chat', request_data, {'error': str(e)}, duration, 'error', log_dir)
                
                return StreamingResponse(
                    chat_with_tracking(),
                    media_type='application/x-ndjson'
                )
            
        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            logger.error(f"Ollama chat request timeout")
            track_metrics(model, 'chat', {}, duration, 'timeout')
            raise HTTPException(status_code=504, detail='Request timeout')
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error proxying chat request: {e}")
            track_metrics(model, 'chat', {}, duration, 'error')
            log_request('chat', request_data, {'error': str(e)}, duration, 'error', log_dir)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post('/api/embeddings')
    async def embeddings(request: Request):
        request_data = await request.json()
        model = request_data.get('model', 'unknown')
        ollama_host = request.app.state.config['ollama_host']
        log_dir = request.app.state.config['log_dir']
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f'{ollama_host}/api/embeddings',
                json=request_data,
                timeout=300
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                
                prompt = request_data.get('prompt', '')
                estimated_tokens = max(1, len(prompt.split()) * 1.3)
                
                metrics_data = {
                    'prompt_eval_count': int(estimated_tokens),
                    'eval_count': 0,
                }
                
                track_metrics(model, 'embeddings', metrics_data, duration, 'success')
                log_request('embeddings', request_data, response_data, duration, 'success', log_dir)
            else:
                track_metrics(model, 'embeddings', {}, duration, 'error')
                log_request('embeddings', request_data, {'error': response.text}, duration, 'error', log_dir)
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type='application/json'
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error proxying embeddings request: {e}")
            track_metrics(model, 'embeddings', {}, duration, 'error')
            log_request('embeddings', request_data, {'error': str(e)}, duration, 'error', log_dir)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/api/tags')
    async def tags(request: Request):
        ollama_host = request.app.state.config['ollama_host']
        
        try:
            response = requests.get(f'{ollama_host}/api/tags', timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                ollama_active_models.set(len(models))
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type='application/json'
            )
        except Exception as e:
            logger.error(f"Error getting tags: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/api/ps')
    async def ps(request: Request):
        ollama_host = request.app.state.config['ollama_host']
        
        try:
            response = requests.get(f'{ollama_host}/api/ps', timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                ollama_active_models.set(len(models))
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type='application/json'
            )
        except Exception as e:
            logger.error(f"Error getting ps: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/metrics')
    async def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )

    @app.get('/health')
    async def health(request: Request):
        ollama_host = request.app.state.config['ollama_host']
        
        try:
            response = requests.get(f'{ollama_host}/api/tags', timeout=5)
            ollama_healthy = response.status_code == 200
        except:
            ollama_healthy = False
        
        status_code = 200 if ollama_healthy else 503
        
        return JSONResponse(
            content={
                'status': 'healthy' if ollama_healthy else 'degraded',
                'ollama_reachable': ollama_healthy,
                'ollama_host': ollama_host
            },
            status_code=status_code
        )

    @app.get('/logs/recent')
    async def recent_logs(request: Request, count: int = 10):
        try:
            log_dir = request.app.state.config['log_dir']
            with open(f'{log_dir}/ollama_requests.jsonl', 'r') as f:
                lines = f.readlines()
                recent = [json.loads(line) for line in lines[-count:]]
            return JSONResponse(content=recent)
        except FileNotFoundError:
            return JSONResponse(content=[])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))