from prometheus_client import Counter, Histogram, Gauge, REGISTRY
from typing import Dict, Any

ollama_tokens_total = Counter(
    'ollama_tokens_total',
    'Total Ollama tokens processed',
    ['model', 'type', 'endpoint']
)

ollama_requests_total = Counter(
    'ollama_requests_total',
    'Total Ollama API requests',
    ['model', 'status', 'endpoint']
)

ollama_request_duration = Histogram(
    'ollama_request_duration_seconds',
    'Ollama API request duration in seconds',
    ['model', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

ollama_eval_duration = Histogram(
    'ollama_eval_duration_seconds',
    'Ollama evaluation duration in seconds',
    ['model', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

ollama_load_duration = Histogram(
    'ollama_load_duration_seconds',
    'Ollama model load duration in seconds',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

ollama_tokens_per_second = Gauge(
    'ollama_tokens_per_second',
    'Tokens generated per second',
    ['model', 'endpoint']
)

ollama_active_models = Gauge(
    'ollama_active_models',
    'Number of active Ollama models loaded in memory'
)


def track_metrics(model: str, endpoint: str, response_data: Dict[str, Any], 
                 duration: float, status: str = 'success') -> None:
    ollama_requests_total.labels(
        model=model,
        status=status,
        endpoint=endpoint
    ).inc()
    
    ollama_request_duration.labels(
        model=model,
        endpoint=endpoint
    ).observe(duration)
    
    if status == 'success' and isinstance(response_data, dict):
        prompt_eval_count = response_data.get('prompt_eval_count', 0)
        eval_count = response_data.get('eval_count', 0)
        total_tokens = prompt_eval_count + eval_count
        
        if prompt_eval_count > 0:
            ollama_tokens_total.labels(
                model=model,
                type='prompt',
                endpoint=endpoint
            ).inc(prompt_eval_count)
        
        if eval_count > 0:
            ollama_tokens_total.labels(
                model=model,
                type='completion',
                endpoint=endpoint
            ).inc(eval_count)
        
        if total_tokens > 0:
            ollama_tokens_total.labels(
                model=model,
                type='total',
                endpoint=endpoint
            ).inc(total_tokens)
        
        eval_duration_ns = response_data.get('eval_duration', 0)
        if eval_duration_ns > 0:
            eval_duration_s = eval_duration_ns / 1e9
            ollama_eval_duration.labels(
                model=model,
                endpoint=endpoint
            ).observe(eval_duration_s)
            
            if eval_count > 0 and eval_duration_s > 0:
                tokens_per_sec = eval_count / eval_duration_s
                ollama_tokens_per_second.labels(
                    model=model,
                    endpoint=endpoint
                ).set(tokens_per_sec)
        
        load_duration_ns = response_data.get('load_duration', 0)
        if load_duration_ns > 0:
            ollama_load_duration.labels(model=model).observe(load_duration_ns / 1e9)


def get_stats() -> Dict[str, Any]:
    stats_data = {
        'total_requests': 0,
        'total_tokens': 0,
        'by_endpoint': {},
        'by_model': {}
    }
    
    for metric in REGISTRY.collect():
        if metric.name == 'ollama_requests_total':
            for sample in metric.samples:
                stats_data['total_requests'] += sample.value
                endpoint = sample.labels.get('endpoint', 'unknown')
                model = sample.labels.get('model', 'unknown')
                
                if endpoint not in stats_data['by_endpoint']:
                    stats_data['by_endpoint'][endpoint] = {'requests': 0, 'tokens': 0}
                stats_data['by_endpoint'][endpoint]['requests'] += sample.value
                
                if model not in stats_data['by_model']:
                    stats_data['by_model'][model] = {'requests': 0, 'tokens': 0}
                stats_data['by_model'][model]['requests'] += sample.value
                
        elif metric.name == 'ollama_tokens_total':
            for sample in metric.samples:
                if sample.labels.get('type') == 'total':
                    stats_data['total_tokens'] += sample.value
                    endpoint = sample.labels.get('endpoint', 'unknown')
                    model = sample.labels.get('model', 'unknown')
                    
                    if endpoint in stats_data['by_endpoint']:
                        stats_data['by_endpoint'][endpoint]['tokens'] += sample.value
                    if model in stats_data['by_model']:
                        stats_data['by_model'][model]['tokens'] += sample.value
    
    return stats_data