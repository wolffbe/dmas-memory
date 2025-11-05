import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


def log_request(endpoint: str, request_data: Dict[str, Any], 
                response_data: Dict[str, Any], duration: float, 
                status: str, log_dir: str = '/app/logs') -> Dict[str, Any]:
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'endpoint': endpoint,
        'model': request_data.get('model', 'unknown'),
        'duration': duration,
        'status': status
    }
    
    if endpoint == 'generate':
        log_entry['prompt'] = request_data.get('prompt', '')[:200]
        log_entry['system'] = request_data.get('system', '')[:100] if request_data.get('system') else None
        log_entry['stream'] = request_data.get('stream', False)
        
        if isinstance(response_data, dict):
            log_entry['response'] = response_data.get('response', '')[:200]
            log_entry['total_duration_ns'] = response_data.get('total_duration', 0)
            log_entry['load_duration_ns'] = response_data.get('load_duration', 0)
            log_entry['prompt_eval_count'] = response_data.get('prompt_eval_count', 0)
            log_entry['prompt_eval_duration_ns'] = response_data.get('prompt_eval_duration', 0)
            log_entry['eval_count'] = response_data.get('eval_count', 0)
            log_entry['eval_duration_ns'] = response_data.get('eval_duration', 0)
    
    elif endpoint == 'chat':
        messages = request_data.get('messages', [])
        log_entry['message_count'] = len(messages)
        log_entry['last_message'] = messages[-1] if messages else None
        log_entry['stream'] = request_data.get('stream', False)
        
        if isinstance(response_data, dict):
            log_entry['response'] = response_data.get('message', {}).get('content', '')[:200]
            log_entry['total_duration_ns'] = response_data.get('total_duration', 0)
            log_entry['load_duration_ns'] = response_data.get('load_duration', 0)
            log_entry['prompt_eval_count'] = response_data.get('prompt_eval_count', 0)
            log_entry['eval_count'] = response_data.get('eval_count', 0)
    
    elif endpoint == 'embeddings':
        log_entry['prompt'] = request_data.get('prompt', '')[:200]
        
        if isinstance(response_data, dict):
            embedding = response_data.get('embedding', [])
            log_entry['embedding_dimensions'] = len(embedding) if embedding else 0
    
    tokens = log_entry.get('eval_count', 0) + log_entry.get('prompt_eval_count', 0)
    logger.info(f"Ollama Request - Endpoint: {endpoint}, Model: {log_entry['model']}, "
                f"Tokens: {tokens}, Duration: {duration:.2f}s, Status: {status}")
    
    try:
        with open(f'{log_dir}/ollama_requests.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logger.error(f"Failed to write log: {e}")
    
    return log_entry