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
    
    if endpoint == 'embeddings':
        log_entry['input'] = request_data.get('input', '')[:200] if isinstance(request_data.get('input'), str) else str(request_data.get('input', ''))[:200]
        log_entry['encoding_format'] = request_data.get('encoding_format', 'float')
        log_entry['dimensions'] = request_data.get('dimensions')
        
        if isinstance(response_data, dict):
            log_entry['usage'] = response_data.get('usage', {})
            log_entry['num_embeddings'] = len(response_data.get('data', []))
    else:
        log_entry['prompt'] = request_data.get('messages') or request_data.get('prompt')
        log_entry['response'] = None
        log_entry['usage'] = {}
        
        if isinstance(response_data, dict):
            if 'choices' in response_data and response_data['choices']:
                choice = response_data['choices'][0]
                if 'message' in choice:
                    log_entry['response'] = choice['message']
                elif 'text' in choice:
                    log_entry['response'] = choice['text']
            
            log_entry['usage'] = response_data.get('usage', {})
    
    tokens = log_entry.get('usage', {}).get('total_tokens', 0)
    logger.info(f"OpenAI Request - Endpoint: {endpoint}, Model: {log_entry['model']}, "
                f"Tokens: {tokens}, Duration: {duration:.2f}s, Status: {status}")
    
    try:
        with open(f'{log_dir}/openai_requests.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logger.error(f"Failed to write log: {e}")
    
    return log_entry