
import json
import os
import pandas as pd
import re
import requests
import shutil
import subprocess
import time

from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")
PROMETHEUS_QUERY_RANGE_URL = PROMETHEUS_URL + "/api/v1/query_range"
PROMETHEUS_QUERY_URL = PROMETHEUS_URL + "/api/v1/query"

TOXIPROXY_URL = os.getenv("TOXIPROXY_URL")

LOCOMO_URL = os.getenv("LOCOMO_URL")

_model = SentenceTransformer('all-MiniLM-L6-v2')

# Default toxic values to avoid NameError when importing
TOXIC_LATENCY = 0
TOXIC_JITTER = 0
TOXIC_BANDWIDTH = 0
TOXIC_SLOW_CLOSE = 0
TOXIC_TIMEOUT = 0
TOXIC_SLICER = 0
TOXIC_LIMIT_DATA = 0
TOXIC_RESET_PEER = 0

def set_env_var(var_name: str, new_value: str, env_file: str = '../.env') -> None:
    env_file = os.path.abspath(env_file)
    content = ''
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            content = f.read()
    pattern = re.compile(rf'^{re.escape(var_name)}=.*$', re.MULTILINE)
    if pattern.search(content):
        content = pattern.sub(f'{var_name}={new_value}', content)
    else:
        if content and not content.endswith('\n'):
            content += '\n'
        content += f'{var_name}={new_value}\n'
    with open(env_file, 'w') as f:
        f.write(content)
        
def ensure_proxy(name, listen, upstream):
    resp = requests.get(f"{TOXIPROXY_URL}/proxies/{name}")
    if resp.status_code == 200:
        return resp.json()
    resp = requests.post(f"{TOXIPROXY_URL}/proxies", json={
        "name": name,
        "listen": listen,
        "upstream": upstream,
    })
    resp.raise_for_status()
    return resp.json()

def clear_toxics(proxy_name):
    resp = requests.get(f"{TOXIPROXY_URL}/proxies/{proxy_name}/toxics")
    if resp.status_code != 200:
        return
    for toxic in resp.json():
        requests.delete(f"{TOXIPROXY_URL}/proxies/{proxy_name}/toxics/{toxic['name']}")
        
def apply_network_profile(profile: dict):
    global TOXIC_LATENCY,  TOXIC_JITTER, TOXIC_BANDWIDTH, TOXIC_SLOW_CLOSE
    global TOXIC_TIMEOUT, TOXIC_SLICER, TOXIC_LIMIT_DATA, TOXIC_RESET_PEER
    
    TOXIC_LATENCY     = int(profile.get("TOXIC_LATENCY", TOXIC_LATENCY))
    TOXIC_JITTER      = int(profile.get("TOXIC_JITTER", TOXIC_JITTER))
    TOXIC_BANDWIDTH   = int(profile.get("TOXIC_BANDWIDTH", TOXIC_BANDWIDTH))
    TOXIC_SLOW_CLOSE  = int(profile.get("TOXIC_SLOW_CLOSE", TOXIC_SLOW_CLOSE))
    TOXIC_TIMEOUT     = int(profile.get("TOXIC_TIMEOUT", TOXIC_TIMEOUT))
    TOXIC_SLICER      = int(profile.get("TOXIC_SLICER", TOXIC_SLICER))
    TOXIC_LIMIT_DATA  = int(profile.get("TOXIC_LIMIT_DATA", TOXIC_LIMIT_DATA))
    TOXIC_RESET_PEER  = int(profile.get("TOXIC_RESET_PEER", TOXIC_RESET_PEER))

    set_env_var("TOXIC_LATENCY", str(TOXIC_LATENCY))
    set_env_var("TOXIC_JITTER", str(TOXIC_JITTER))
    set_env_var("TOXIC_BANDWIDTH", str(TOXIC_BANDWIDTH))
    set_env_var("TOXIC_SLOW_CLOSE", str(TOXIC_SLOW_CLOSE))
    set_env_var("TOXIC_TIMEOUT", str(TOXIC_TIMEOUT))
    set_env_var("TOXIC_SLICER", str(TOXIC_SLICER))
    set_env_var("TOXIC_LIMIT_DATA", str(TOXIC_LIMIT_DATA))
    set_env_var("TOXIC_RESET_PEER", str(TOXIC_RESET_PEER))

    ensure_proxy("memory-proxy",    "0.0.0.0:18005", "memory:8002")
    ensure_proxy("responder-proxy", "0.0.0.0:18006", "responder:8003")
    ensure_proxy("locomo-proxy",    "0.0.0.0:18007", "locomo:8000")
    
    for name in ["memory-proxy", "responder-proxy", "locomo-proxy"]:
        clear_toxics(name)

        if TOXIC_LATENCY > 0:
            resp = requests.post(f"{TOXIPROXY_URL}/proxies/{name}/toxics", json={
                "name": "latency",
                "type": "latency",
                "stream": "downstream",
                "attributes": {
                    "latency": TOXIC_LATENCY,
                    "jitter": TOXIC_JITTER if TOXIC_JITTER > 0 else 0,
                }
            })
            resp.raise_for_status()

        if TOXIC_BANDWIDTH > 0:
            resp = requests.post(f"{TOXIPROXY_URL}/proxies/{name}/toxics", json={
                "name": "bandwidth",
                "type": "bandwidth",
                "stream": "downstream",
                "attributes": {
                    "rate": TOXIC_BANDWIDTH * 1024,
                }
            })
            resp.raise_for_status()
            
def verify_memory_backend():
    try:
        memory_backend = subprocess.check_output(
            ["docker", "exec", "memory", "printenv", "MEMORY_BACKEND"],
            text=True
        ).strip()
    except subprocess.CalledProcessError:
        print(f"Error: Cannot get MEMORY_BACKEND from memory container. Make sure the container exists and is running.")
        return
    
    expected_backend = os.environ.get("MEMORY_BACKEND", "").strip()
    if memory_backend != expected_backend:
        raise ValueError(f"MEMORY_BACKEND mismatch: container has '{memory_backend}', expected '{expected_backend}'")
    
    print(f"MEMORY_BACKEND correctly set to '{memory_backend}' in memory container")

def check_toxics(PROFILE, TOXIPROXY_URL):
    try:
        r = requests.get(f"{TOXIPROXY_URL}/proxies").json()
        for name, proxy in r.items():
            for toxic in proxy["toxics"]:
                if toxic["type"] == "latency":
                    assert toxic["attributes"]["latency"] == PROFILE["TOXIC_LATENCY"], f"{name} latency mismatch"
                    assert toxic["attributes"]["jitter"] == PROFILE["TOXIC_JITTER"], f"{name} jitter mismatch"
                if toxic["type"] == "bandwidth":
                    assert toxic["attributes"]["rate"] == int(PROFILE["TOXIC_BANDWIDTH"]) * 1024, f"{name} bandwidth mismatch"
        print("All toxics are correctly set.")
    except Exception as e:
        print(f"Error: {e}")
        
def query_range(query, start, end, step="1s"):
    params = {"query": query, "start": start, "end": end, "step": step}
    r = requests.get(PROMETHEUS_QUERY_RANGE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    total = 0.0
    for result in data['data']['result']:
        for v in result['values']:
            total += float(v[1])
    return total

def get_instant_tokens(prometheus_url=PROMETHEUS_QUERY_URL):
    query = 'sum(openai_tokens_total{type="total"})'
    r = requests.get(prometheus_url, params={"query": query}, timeout=10)
    if r.status_code == 200:
        data = r.json()
        if data['status'] == 'success' and data['data']['result']:
            return float(data['data']['result'][0]['value'][1])
    return 0

def normalize_model_name(model_name):
    import re
    normalized = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', model_name)
    return normalized

def get_openai_cost(start, end, pricing_file="openai_prices.json", tier="standard"):
    try:
        with open(pricing_file, 'r') as f:
            pricing_data = json.load(f)
        
        pricing_list = pricing_data['pricing']['text_tokens'][tier]
        text_pricing = {item['model']: {'input': item['input'], 'output': item['output']} for item in pricing_list}
        
        embedding_pricing = {item['model']: item['cost'] for item in pricing_data['pricing']['embeddings']}
        
        total_cost = 0.0
        
        for token_type in ['prompt', 'completion']:
            query = f'sum by (model) (openai_tokens_total{{type="{token_type}",endpoint="chat.completions"}})'
            
            r_before = requests.get(PROMETHEUS_QUERY_URL, params={"query": query, "time": start}, timeout=10)
            r_after = requests.get(PROMETHEUS_QUERY_URL, params={"query": query, "time": end + 2}, timeout=10)
            
            if r_before.status_code == 200 and r_after.status_code == 200:
                data_before = r_before.json()
                data_after = r_after.json()
                
                before_values = {}
                if data_before['status'] == 'success':
                    for result in data_before['data']['result']:
                        model = result['metric'].get('model', 'unknown')
                        before_values[model] = float(result['value'][1])
                
                if data_after['status'] == 'success':
                    for result in data_after['data']['result']:
                        model = result['metric'].get('model', 'unknown')
                        after_value = float(result['value'][1])
                        before_value = before_values.get(model, 0)
                        tokens = after_value - before_value
                        
                        if tokens > 0:
                            normalized_model = normalize_model_name(model)
                            
                            if model in text_pricing:
                                pricing_model = model
                            elif normalized_model in text_pricing:
                                pricing_model = normalized_model
                            else:
                                continue
                            
                            price_per_million = text_pricing[pricing_model]['input'] if token_type == 'prompt' else text_pricing[pricing_model]['output']
                            if price_per_million:
                                cost = (tokens / 1_000_000) * price_per_million
                                total_cost += cost
        
        query = f'sum by (model) (openai_tokens_total{{endpoint="embeddings"}})'
        
        r_before = requests.get(PROMETHEUS_QUERY_URL, params={"query": query, "time": start}, timeout=10)
        r_after = requests.get(PROMETHEUS_QUERY_URL, params={"query": query, "time": end + 2}, timeout=10)
        
        if r_before.status_code == 200 and r_after.status_code == 200:
            data_before = r_before.json()
            data_after = r_after.json()
            
            before_values = {}
            if data_before['status'] == 'success':
                for result in data_before['data']['result']:
                    model = result['metric'].get('model', 'unknown')
                    before_values[model] = float(result['value'][1])
            
            if data_after['status'] == 'success':
                for result in data_after['data']['result']:
                    model = result['metric'].get('model', 'unknown')
                    after_value = float(result['value'][1])
                    before_value = before_values.get(model, 0)
                    tokens = after_value - before_value
                    
                    if tokens > 0 and model in embedding_pricing:
                        cost = (tokens / 1_000_000) * embedding_pricing[model]
                        total_cost += cost
        
        return total_cost
    except Exception as e:
        print(f"Error calculating OpenAI cost: {e}")
        import traceback
        traceback.print_exc()
        return 0

def measure_function(func, *args, min_duration=15, **kwargs):
    tokens_before = get_instant_tokens()

    start = time.time()
    print(f"Starting function '{func.__name__}' at {start}")
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Function '{func.__name__}' completed at {end}")
    actual_duration = end - start

    duration_s = max(int(actual_duration), min_duration)
    if actual_duration < min_duration:
        time.sleep(min_duration - actual_duration)
        end = time.time()

    time.sleep(2)
    
    tokens_after = get_instant_tokens()

    cpu_edge = query_range(f'sum(increase(docker_container_cpu_usage_total{{group="edge"}}[{duration_s}s]))', start, end)
    cpu_cloud = query_range(f'sum(increase(docker_container_cpu_usage_total{{group="cloud"}}[{duration_s}s]))', start, end)
    ram_edge = query_range(f'avg_over_time(docker_container_mem_usage{{group="edge"}}[{duration_s}s])', start, end)
    ram_cloud = query_range(f'avg_over_time(docker_container_mem_usage{{group="cloud"}}[{duration_s}s])', start, end)
    
    disk_edge_read = query_range(f'sum(increase(docker_container_blkio_io_service_bytes_recursive_read{{group="edge"}}[{duration_s}s]))', start, end)
    disk_edge_write = query_range(f'sum(increase(docker_container_blkio_io_service_bytes_recursive_write{{group="edge"}}[{duration_s}s]))', start, end)
    disk_edge = disk_edge_read + disk_edge_write
    
    disk_cloud_read = query_range(f'sum(increase(docker_container_blkio_io_service_bytes_recursive_read{{group="cloud"}}[{duration_s}s]))', start, end)
    disk_cloud_write = query_range(f'sum(increase(docker_container_blkio_io_service_bytes_recursive_write{{group="cloud"}}[{duration_s}s]))', start, end)
    disk_cloud = disk_cloud_read + disk_cloud_write
    
    network_edge = query_range(f'sum(increase(docker_container_net_rx_bytes{{group="edge"}}[{duration_s}s]))', start, end)
    network_cloud = query_range(f'sum(increase(docker_container_net_rx_bytes{{group="cloud"}}[{duration_s}s]))', start, end)
    
    tokens = int(tokens_after - tokens_before)
    
    token_cost = get_openai_cost(start, end)

    metrics = {
        "cpu_edge_ns": cpu_edge,
        "cpu_cloud_ns": cpu_cloud,
        "ram_edge_bytes": ram_edge,
        "ram_cloud_bytes": ram_cloud,
        "disk_edge_bytes": disk_edge,
        "disk_cloud_bytes": disk_cloud,
        "network_edge_bytes": network_edge,
        "network_cloud_bytes": network_cloud,
        "openai_tokens": tokens,
        "openai_cost_usd": token_cost,
        "api_latency_s": actual_duration,
        "metric_window_s": duration_s
    }

    return {"result": result, "metrics": metrics}

def load_session(session_id, conversation_index):
    requests.post(f"{LOCOMO_URL}/conversation/load/{conversation_index}/session/{session_id}")

def load_memories(sessions, conversation_index, MEMORY_BACKEND):
    memories = []
    num_sessions = max(1, sessions)
    
    for i in range(1, num_sessions + 1):
        print(f"Loading session: {i}")
        output = measure_function(load_session, i, conversation_index)
        row = {
            "memory": MEMORY_BACKEND,
            "time": time.time(),
            "conversation_index": conversation_index,
            "session_id": i,
            **output.get("metrics", {}),
            "toxic_latency": os.getenv("TOXIC_LATENCY", ""),
            "toxic_jitter": os.getenv("TOXIC_JITTER", ""),
            "toxic_bandwidth": os.getenv("TOXIC_BANDWIDTH", ""),
            "toxic_slow_close": os.getenv("TOXIC_SLOW_CLOSE", ""),
            "toxic_timeout": os.getenv("TOXIC_TIMEOUT", ""),
            "toxic_slicer": os.getenv("TOXIC_SLICER", ""),
            "toxic_limit_data": os.getenv("TOXIC_LIMIT_DATA", ""),
            "toxic_reset_peer": os.getenv("TOXIC_RESET_PEER", "")
        }
        print(row)
        memories.append(row)
    return memories

def export_loaded_memory_metrics(memories, MEMORY_BACKEND, conversation_index):
    if not memories:
        print("No memories to save.")
        return
    
    df = pd.DataFrame(memories)
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    is_constrained = any(
        int(os.getenv(var, 0)) > 0 for var in [
            "TOXIC_LATENCY", "TOXIC_JITTER", "TOXIC_BANDWIDTH",
            "TOXIC_SLOW_CLOSE", "TOXIC_TIMEOUT", "TOXIC_SLICER",
            "TOXIC_LIMIT_DATA", "TOXIC_RESET_PEER"
        ]
    )
    
    output_file = f"{MEMORY_BACKEND}_conv_{conversation_index}_sessions{'_constrained' if is_constrained else ''}.csv"
    output_path = os.path.join(output_dir, output_file)
    
    if os.path.exists(output_path):
        backup_file = f"{MEMORY_BACKEND}_conv_{conversation_index}_sessions{'_constrained' if is_constrained else ''}_backup.csv"
        backup_path = os.path.join(output_dir, backup_file)
        shutil.move(output_path, backup_path)
        print(f"Existing file backed up to {backup_path}")
    
    df.to_csv(output_path, index=False)
    print(f"Saved new results to {output_path}")
    
def check_similarity_string(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def check_similarity_semantic(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    emb1 = _model.encode(text1, convert_to_tensor=True)
    emb2 = _model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())

def run_qa(questions, MEMORY_BACKEND, CONVERSATION_INDEX, PROFILE):
    qa = []
    total_questions = len(questions.get("questions", []))
    
    def ask(question):
        resp = requests.post(f"{LOCOMO_URL}/ask", params={"question": question})
        return resp.json().get("answer")
    
    for idx, question in enumerate(questions.get("questions", []), start=1):
        print(f"[{idx}/{total_questions}] Asking question: {question.get('question')}")
        output = measure_function(ask, question.get("question"))
        print(f"[{idx}/{total_questions}] Answer:", output["result"])
        
        answer_received = output["result"]
        metrics = output.get("metrics", {})
        
        similarity_string = check_similarity_string(question.get("answer"), answer_received)
        similarity_semantic = check_similarity_semantic(question.get("answer"), answer_received)
        
        row = {
            "memory": MEMORY_BACKEND,
            "time": time.time(),
            "conversation_index": CONVERSATION_INDEX,
            "question": question.get("question"),
            "answer_actual": question.get("answer"),
            "answer_received": answer_received,
            "similarity_string": similarity_string,
            "similarity_semantic": similarity_semantic,
            "similarity": (similarity_string + similarity_semantic) / 2,
            **metrics,
            "toxic_latency": PROFILE["TOXIC_LATENCY"],
            "toxic_jitter": PROFILE["TOXIC_JITTER"],
            "toxic_bandwidth": PROFILE["TOXIC_BANDWIDTH"],
            "toxic_slow_close": PROFILE["TOXIC_SLOW_CLOSE"],
            "toxic_timeout": PROFILE["TOXIC_TIMEOUT"],
            "toxic_slicer": PROFILE["TOXIC_SLICER"],
            "toxic_limit_data": PROFILE["TOXIC_LIMIT_DATA"],
            "toxic_reset_peer": PROFILE["TOXIC_RESET_PEER"],
        }
        
        print(row)
        qa.append(row)
    
    return qa


def export_qa(qa, MEMORY_BACKEND, conversation_index):
    if not qa:
        print("No QA results to save.")
        return
    
    df = pd.DataFrame(qa)
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    is_constrained = any(
        int(os.getenv(var, 0)) > 0 for var in [
            "TOXIC_LATENCY", "TOXIC_JITTER", "TOXIC_BANDWIDTH",
            "TOXIC_SLOW_CLOSE", "TOXIC_TIMEOUT", "TOXIC_SLICER",
            "TOXIC_LIMIT_DATA", "TOXIC_RESET_PEER"
        ]
    )
    
    output_file = f"{MEMORY_BACKEND}_conv_{conversation_index}_qa{'_constrained' if is_constrained else ''}.csv"
    output_path = os.path.join(output_dir, output_file)
    
    if os.path.exists(output_path):
        backup_file = f"{MEMORY_BACKEND}_conv_{conversation_index}_qa{'_constrained' if is_constrained else ''}_backup.csv"
        backup_path = os.path.join(output_dir, backup_file)
        shutil.move(output_path, backup_path)
        print(f"Existing file backed up to {backup_path}")
    
    df.to_csv(output_path, index=False)
    print(f"Saved new QA results to {output_path}")