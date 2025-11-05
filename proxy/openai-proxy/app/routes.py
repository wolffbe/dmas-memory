import logging
import time
import json
import requests
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from metrics import track_metrics, get_stats
from utils import log_request

logger = logging.getLogger(__name__)

def register_routes(app: FastAPI) -> None:
    
    @app.post('/v1/chat/completions')
    async def chat_completions(request: Request):
        request_data = await request.json()
        model = request_data.get('model', 'unknown')
        messages = request_data.get('messages', [])
        
        if messages:
            first_msg = messages[0].get('content', '')[:400] if messages else ''
            logger.info(f"[OpenAI] {model} chat: {first_msg}...")
        else:
            logger.info(f"[OpenAI] {model}: {json.dumps(messages)}")
        
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not api_key:
            raise HTTPException(status_code=401, detail='No API key provided')
        
        openai_base_url = request.app.state.config['openai_base_url']
        log_dir = request.app.state.config['log_dir']
        start_time = time.time()
        
        try:
            response = requests.post(
                f'{openai_base_url}/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json=request_data,
                timeout=300
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                usage = response_data.get('usage', {})
                
                track_metrics(model, 'chat.completions', usage, duration, 'success')
                log_request('chat.completions', request_data, response_data, duration, 'success', log_dir)
            else:
                track_metrics(model, 'chat.completions', {}, duration, 'error')
                log_request('chat.completions', request_data, {'error': response.text}, duration, 'error', log_dir)
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type='application/json'
            )
            
        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            logger.error(f"OpenAI request timeout after {duration:.2f}s")
            track_metrics(model, 'chat.completions', {}, duration, 'timeout')
            raise HTTPException(status_code=504, detail='Request timeout')
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error proxying request: {e}")
            track_metrics(model, 'chat.completions', {}, duration, 'error')
            log_request('chat.completions', request_data, {'error': str(e)}, duration, 'error', log_dir)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post('/v1/completions')
    async def completions(request: Request):
        request_data = await request.json()
        model = request_data.get('model', 'unknown')
        prompt = request_data.get('prompt', '')
        
        prompt_preview = str(prompt)[:400] if prompt else ''
        logger.info(f"[OpenAI] {model} completion: {prompt_preview}...")
        
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not api_key:
            raise HTTPException(status_code=401, detail='No API key provided')
        
        openai_base_url = request.app.state.config['openai_base_url']
        log_dir = request.app.state.config['log_dir']
        start_time = time.time()
        
        try:
            response = requests.post(
                f'{openai_base_url}/v1/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json=request_data,
                timeout=300
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                usage = response_data.get('usage', {})
                
                track_metrics(model, 'completions', usage, duration, 'success')
                log_request('completions', request_data, response_data, duration, 'success', log_dir)
            else:
                track_metrics(model, 'completions', {}, duration, 'error')
                log_request('completions', request_data, {'error': response.text}, duration, 'error', log_dir)
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type='application/json'
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error proxying request: {e}")
            track_metrics(model, 'completions', {}, duration, 'error')
            log_request('completions', request_data, {'error': str(e)}, duration, 'error', log_dir)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post('/v1/embeddings')
    async def embeddings(request: Request):
        request_data = await request.json()
        model = request_data.get('model', 'unknown')
        input_text = request_data.get('input', '')
        
        input_preview = str(input_text)[:400] if input_text else ''
        logger.info(f"[OpenAI] {model} embedding: {input_preview}...")
        
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not api_key:
            raise HTTPException(status_code=401, detail='No API key provided')
        
        openai_base_url = request.app.state.config['openai_base_url']
        log_dir = request.app.state.config['log_dir']
        start_time = time.time()
        
        try:
            response = requests.post(
                f'{openai_base_url}/v1/embeddings',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json=request_data,
                timeout=300
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                usage = response_data.get('usage', {})
                
                track_metrics(model, 'embeddings', usage, duration, 'success')
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

    @app.post('/v1/responses')
    async def responses(request: Request):
        request_data = await request.json()
        model = request_data.get('model', 'unknown')
        
        logger.info(f"[OpenAI] {model} responses - Keys: {list(request_data.keys())}")
        
        messages = request_data.get('messages', [])
        if messages and isinstance(messages, list) and len(messages) > 0:
            first_msg = messages[0].get('content', '')[:400] if isinstance(messages[0], dict) else str(messages[0])[:400]
            logger.info(f"[OpenAI] {model}: {first_msg}...")
        else:
            prompt = request_data.get('prompt', request_data.get('input', ''))
            if prompt:
                logger.info(f"[OpenAI] {model}: {str(prompt)[:400]}...")
            else:
                logger.info(f"[OpenAI] {model}: (no content found)")
        
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not api_key:
            raise HTTPException(status_code=401, detail='No API key provided')
        
        openai_base_url = request.app.state.config['openai_base_url']
        log_dir = request.app.state.config['log_dir']
        start_time = time.time()
        
        try:
            response = requests.post(
                f'{openai_base_url}/v1/responses',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json=request_data,
                timeout=300
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                usage = response_data.get('usage', {})
                
                track_metrics(model, 'responses', usage, duration, 'success')
                log_request('responses', request_data, response_data, duration, 'success', log_dir)
            else:
                track_metrics(model, 'responses', {}, duration, 'error')
                log_request('responses', request_data, {'error': response.text}, duration, 'error', log_dir)
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type='application/json'
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error proxying responses request: {e}")
            track_metrics(model, 'responses', {}, duration, 'error')
            log_request('responses', request_data, {'error': str(e)}, duration, 'error', log_dir)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/metrics')
    async def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )

    @app.get('/health')
    async def health():
        return JSONResponse(content={'status': 'healthy'})