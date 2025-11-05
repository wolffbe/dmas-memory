from prometheus_client import Counter, Histogram, REGISTRY
from typing import Dict, Any

openai_tokens_total = Counter(
    'openai_tokens_total',
    'Total OpenAI tokens used',
    ['model', 'type', 'endpoint']
)

openai_requests_total = Counter(
    'openai_requests_total',
    'Total OpenAI API requests',
    ['model', 'status', 'endpoint']
)

openai_request_duration = Histogram(
    'openai_request_duration_seconds',
    'OpenAI API request duration',
    ['model', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

def track_metrics(model: str, endpoint: str, usage: Dict[str, Any], 
                 duration: float, status: str = 'success') -> None:
    openai_requests_total.labels(
        model=model,
        status=status,
        endpoint=endpoint
    ).inc()
    
    openai_request_duration.labels(
        model=model,
        endpoint=endpoint
    ).observe(duration)
    
    if status == 'success' and usage:
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        if prompt_tokens > 0:
            openai_tokens_total.labels(
                model=model,
                type='prompt',
                endpoint=endpoint
            ).inc(prompt_tokens)
        
        if completion_tokens > 0:
            openai_tokens_total.labels(
                model=model,
                type='completion',
                endpoint=endpoint
            ).inc(completion_tokens)
        
        if total_tokens > 0:
            openai_tokens_total.labels(
                model=model,
                type='total',
                endpoint=endpoint
            ).inc(total_tokens)

def get_stats() -> Dict[str, Any]:
    stats_data = {
        'total_requests': 0,
        'total_tokens': 0,
        'by_endpoint': {}
    }
    
    for metric in REGISTRY.collect():
        if metric.name == 'openai_requests_total':
            for sample in metric.samples:
                stats_data['total_requests'] += sample.value
                endpoint = sample.labels.get('endpoint', 'unknown')
                
                if endpoint not in stats_data['by_endpoint']:
                    stats_data['by_endpoint'][endpoint] = {'requests': 0, 'tokens': 0, 'cost': 0}
                stats_data['by_endpoint'][endpoint]['requests'] += sample.value
                
        elif metric.name == 'openai_tokens_total':
            for sample in metric.samples:
                if sample.labels.get('type') == 'total':
                    stats_data['total_tokens'] += sample.value
                    endpoint = sample.labels.get('endpoint', 'unknown')
                    
                    if endpoint in stats_data['by_endpoint']:
                        stats_data['by_endpoint'][endpoint]['tokens'] += sample.value
                    
    return stats_data