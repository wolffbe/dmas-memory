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

print("\n" + "="*60)
print("LOADING ENVIRONMENT VARIABLES")
print("="*60)

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")
print(f"PROMETHEUS_URL: {PROMETHEUS_URL if PROMETHEUS_URL else 'NOT SET'}")

if PROMETHEUS_URL:
    PROMETHEUS_QUERY_RANGE_URL = PROMETHEUS_URL + "/api/v1/query_range"
    PROMETHEUS_QUERY_URL = PROMETHEUS_URL + "/api/v1/query"
else:
    PROMETHEUS_QUERY_RANGE_URL = None
    PROMETHEUS_QUERY_URL = None
    print("WARNING: PROMETHEUS_URL not set - query URLs will be None")

TOXIPROXY_URL = os.getenv("TOXIPROXY_URL")
print(f"TOXIPROXY_URL: {TOXIPROXY_URL if TOXIPROXY_URL else 'NOT SET'}")

COORDINATOR_URL = os.getenv("COORDINATOR_URL")
print(f"COORDINATOR_URL: {COORDINATOR_URL if COORDINATOR_URL else 'NOT SET'}")

print("="*60 + "\n")

_model = SentenceTransformer('all-MiniLM-L6-v2')

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
    print(f"Setting {var_name}={new_value} in {env_file}")
    
    content = ''
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            content = f.read()
        print(f"Found existing .env file")
    else:
        print(f"Creating new .env file")
        
    pattern = re.compile(rf'^{re.escape(var_name)}=.*$', re.MULTILINE)
    if pattern.search(content):
        content = pattern.sub(f'{var_name}={new_value}', content)
        print(f"Updated existing variable {var_name}")
    else:
        if content and not content.endswith('\n'):
            content += '\n'
        content += f'{var_name}={new_value}\n'
        print(f"Added new variable {var_name}")
        
    with open(env_file, 'w') as f:
        f.write(content)
    print(f"Successfully wrote to {env_file}")
        
def ensure_proxy(name, listen, upstream):
    print(f"Ensuring proxy '{name}' exists (listen: {listen}, upstream: {upstream})")
    resp = requests.get(f"{TOXIPROXY_URL}/proxies/{name}")
    if resp.status_code == 200:
        print(f"Proxy '{name}' already exists")
        return resp.json()
    
    print(f"Creating new proxy '{name}'")
    resp = requests.post(f"{TOXIPROXY_URL}/proxies", json={
        "name": name,
        "listen": listen,
        "upstream": upstream,
    })
    resp.raise_for_status()
    print(f"Successfully created proxy '{name}'")
    return resp.json()

def clear_toxics(proxy_name):
    print(f"Clearing toxics for proxy '{proxy_name}'")
    resp = requests.get(f"{TOXIPROXY_URL}/proxies/{proxy_name}/toxics")
    if resp.status_code != 200:
        print(f"No toxics found for '{proxy_name}'")
        return
    
    toxics = resp.json()
    print(f"Found {len(toxics)} toxic(s) to clear")
    for toxic in toxics:
        requests.delete(f"{TOXIPROXY_URL}/proxies/{proxy_name}/toxics/{toxic['name']}")
        print(f"Deleted toxic '{toxic['name']}'")
        
def apply_network_profile(profile: dict):
    global TOXIC_LATENCY,  TOXIC_JITTER, TOXIC_BANDWIDTH, TOXIC_SLOW_CLOSE
    global TOXIC_TIMEOUT, TOXIC_SLICER, TOXIC_LIMIT_DATA, TOXIC_RESET_PEER
    
    print("\n" + "="*60)
    print("APPLYING NETWORK PROFILE")
    print("="*60)
    
    TOXIC_LATENCY     = int(profile.get("TOXIC_LATENCY", TOXIC_LATENCY))
    TOXIC_JITTER      = int(profile.get("TOXIC_JITTER", TOXIC_JITTER))
    TOXIC_BANDWIDTH   = int(profile.get("TOXIC_BANDWIDTH", TOXIC_BANDWIDTH))
    TOXIC_SLOW_CLOSE  = int(profile.get("TOXIC_SLOW_CLOSE", TOXIC_SLOW_CLOSE))
    TOXIC_TIMEOUT     = int(profile.get("TOXIC_TIMEOUT", TOXIC_TIMEOUT))
    TOXIC_SLICER      = int(profile.get("TOXIC_SLICER", TOXIC_SLICER))
    TOXIC_LIMIT_DATA  = int(profile.get("TOXIC_LIMIT_DATA", TOXIC_LIMIT_DATA))
    TOXIC_RESET_PEER  = int(profile.get("TOXIC_RESET_PEER", TOXIC_RESET_PEER))

    print(f"TOXIC_LATENCY: {TOXIC_LATENCY}")
    print(f"TOXIC_JITTER: {TOXIC_JITTER}")
    print(f"TOXIC_BANDWIDTH: {TOXIC_BANDWIDTH}")
    print(f"TOXIC_SLOW_CLOSE: {TOXIC_SLOW_CLOSE}")
    print(f"TOXIC_TIMEOUT: {TOXIC_TIMEOUT}")
    print(f"TOXIC_SLICER: {TOXIC_SLICER}")
    print(f"TOXIC_LIMIT_DATA: {TOXIC_LIMIT_DATA}")
    print(f"TOXIC_RESET_PEER: {TOXIC_RESET_PEER}")

    set_env_var("TOXIC_LATENCY", str(TOXIC_LATENCY))
    set_env_var("TOXIC_JITTER", str(TOXIC_JITTER))
    set_env_var("TOXIC_BANDWIDTH", str(TOXIC_BANDWIDTH))
    set_env_var("TOXIC_SLOW_CLOSE", str(TOXIC_SLOW_CLOSE))
    set_env_var("TOXIC_TIMEOUT", str(TOXIC_TIMEOUT))
    set_env_var("TOXIC_SLICER", str(TOXIC_SLICER))
    set_env_var("TOXIC_LIMIT_DATA", str(TOXIC_LIMIT_DATA))
    set_env_var("TOXIC_RESET_PEER", str(TOXIC_RESET_PEER))

    print("Ensuring proxies exist...")
    ensure_proxy("memory-proxy",    "0.0.0.0:18005", "memory:8002")
    ensure_proxy("responder-proxy", "0.0.0.0:18006", "responder:8003")
    ensure_proxy("locomo-proxy",    "0.0.0.0:18007", "locomo:8000")
    
    for name in ["memory-proxy", "responder-proxy", "locomo-proxy"]:
        clear_toxics(name)

        if TOXIC_LATENCY > 0:
            print(f"Adding latency toxic to '{name}': {TOXIC_LATENCY}ms (jitter: {TOXIC_JITTER}ms)")
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
            print(f"Adding bandwidth toxic to '{name}': {TOXIC_BANDWIDTH} KB/s")
            resp = requests.post(f"{TOXIPROXY_URL}/proxies/{name}/toxics", json={
                "name": "bandwidth",
                "type": "bandwidth",
                "stream": "downstream",
                "attributes": {
                    "rate": TOXIC_BANDWIDTH * 1024,
                }
            })
            resp.raise_for_status()
    
    print("="*60 + "\n")
            
def verify_memory_backend():
    print("\nChecking MEMORY_BACKEND in container...")
    try:
        memory_backend = subprocess.check_output(
            ["docker", "exec", "memory", "printenv", "MEMORY_BACKEND"],
            text=True
        ).strip()
        print(f"Container MEMORY_BACKEND: {memory_backend}")
    except subprocess.CalledProcessError as e:
        print(f"Cannot get MEMORY_BACKEND from memory container: {e}")
        print(f"Make sure the container exists and is running.")
        return
    
    expected_backend = os.environ.get("MEMORY_BACKEND", "").strip()
    print(f"Expected MEMORY_BACKEND: {expected_backend}")
    
    if memory_backend != expected_backend:
        raise ValueError(f"MEMORY_BACKEND mismatch: container has '{memory_backend}', expected '{expected_backend}'")
    
    print(f"✓ MEMORY_BACKEND correctly set to '{memory_backend}' in memory container\n")

def check_toxics(PROFILE, TOXIPROXY_URL):
    print("\nVerifying toxic configuration...")
    try:
        r = requests.get(f"{TOXIPROXY_URL}/proxies").json()
        for name, proxy in r.items():
            print(f"Checking proxy '{name}'...")
            for toxic in proxy["toxics"]:
                if toxic["type"] == "latency":
                    assert toxic["attributes"]["latency"] == PROFILE["TOXIC_LATENCY"], f"{name} latency mismatch"
                    print(f"  ✓ Latency: {toxic['attributes']['latency']}ms")
                    assert toxic["attributes"]["jitter"] == PROFILE["TOXIC_JITTER"], f"{name} jitter mismatch"
                    print(f"  ✓ Jitter: {toxic['attributes']['jitter']}ms")
                if toxic["type"] == "bandwidth":
                    assert toxic["attributes"]["rate"] == int(PROFILE["TOXIC_BANDWIDTH"]) * 1024, f"{name} bandwidth mismatch"
                    print(f"  ✓ Bandwidth: {toxic['attributes']['rate']/1024} KB/s")
        print("✓ All toxics are correctly set.\n")
    except Exception as e:
        print(f"Toxic check failed: {e}\n")
        
def query_range(query, start, end, step="1s"):
    print(f"Prometheus range query: {query[:80]}...")
    params = {"query": query, "start": start, "end": end, "step": step}
    r = requests.get(PROMETHEUS_QUERY_RANGE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    total = 0.0
    for result in data['data']['result']:
        for v in result['values']:
            total += float(v[1])
    print(f"Result: {total}")
    return total

def get_instant_tokens(prometheus_url=PROMETHEUS_QUERY_URL):
    query = 'sum(openai_tokens_total{type="total"})'
    r = requests.get(prometheus_url, params={"query": query}, timeout=10)
    if r.status_code == 200:
        data = r.json()
        if data['status'] == 'success' and data['data']['result']:
            tokens = float(data['data']['result'][0]['value'][1])
            print(f"Current token count: {tokens}")
            return tokens
    print(f"No token data available")
    return 0

def normalize_model_name(model_name):
    import re
    normalized = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', model_name)
    return normalized

def get_openai_cost(start, end, pricing_file="openai_prices.json", tier="standard"):
    print(f"Calculating OpenAI cost from {start} to {end}...")
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
                                print(f"{model} ({token_type}): {tokens} tokens = ${cost:.4f}")
        
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
                        print(f"{model} (embeddings): {tokens} tokens = ${cost:.4f}")
        
        print(f"Total cost: ${total_cost:.4f}\n")
        return total_cost
    except Exception as e:
        print(f"Error calculating OpenAI cost: {e}")
        import traceback
        traceback.print_exc()
        return 0

def measure_function(func, *args, min_duration=15, **kwargs):
    print("\n" + "-"*60)
    print(f"Starting measurement for function '{func.__name__}'")
    print("-"*60)
    
    tokens_before = get_instant_tokens()
    print(f"COORDINATOR_URL: {COORDINATOR_URL}")
    start = time.time()
    print(f"Start time: {start}")
    
    result = func(*args, **kwargs)
    
    end = time.time()
    print(f"End time: {end}")
    actual_duration = end - start
    print(f"Actual duration: {actual_duration:.2f}s")

    duration_s = max(int(actual_duration), min_duration)
    if actual_duration < min_duration:
        wait_time = min_duration - actual_duration
        print(f"Waiting {wait_time:.2f}s to meet minimum duration of {min_duration}s")
        time.sleep(wait_time)
        end = time.time()

    print(f"Sleeping 2s before collecting metrics...")
    time.sleep(2)
    
    tokens_after = get_instant_tokens()

    print(f"Collecting resource metrics (window: {duration_s}s)...")
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
    print(f"Tokens used: {tokens}")
    
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

    print(f"Measurement complete")
    print("-"*60 + "\n")
    
    return {"result": result, "metrics": metrics}

def load_session(session_id, conversation_index):
    print(f"Loading session {session_id} for conversation {conversation_index}")
    requests.post(f"{COORDINATOR_URL}/conversation/load/{conversation_index}/session/{session_id}")

def load_memories(sessions, conversation_index, MEMORY_BACKEND):
    print(f"\nLOADING MEMORIES: {sessions} session(s)")
    print(f"Backend: {MEMORY_BACKEND}, Conversation: {conversation_index}")
    print("="*60 + "\n")
    
    memories = []
    num_sessions = max(1, sessions)
    
    for i in range(1, num_sessions + 1):
        print(f"\nLoading session {i}/{num_sessions}")
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
        print(f"Session {i} metrics: {row}")
        memories.append(row)
    
    print("\n" + "="*60)
    print(f"COMPLETED LOADING {len(memories)} SESSION(S)")
    print("="*60 + "\n")
    return memories

def export_loaded_memory_metrics(memories, MEMORY_BACKEND, conversation_index):
    if not memories:
        print("No memories to save.")
        return
    
    print(f"Exporting {len(memories)} memory metric(s)...")
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
    print(f"Constrained mode: {is_constrained}")
    
    output_file = f"{MEMORY_BACKEND}_conv_{conversation_index}_sessions{'_constrained' if is_constrained else ''}.csv"
    output_path = os.path.join(output_dir, output_file)
    
    if os.path.exists(output_path):
        backup_file = f"{MEMORY_BACKEND}_conv_{conversation_index}_sessions{'_constrained' if is_constrained else ''}_backup.csv"
        backup_path = os.path.join(output_dir, backup_file)
        shutil.move(output_path, backup_path)
        print(f"Existing file backed up to {backup_path}")
    
    df.to_csv(output_path, index=False)
    print(f"✓ Saved new results to {output_path}\n")
    
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
    print("\n" + "="*60)
    print("RUNNING Q&A SESSION")
    print("="*60)
    
    qa = []
    total_questions = len(questions.get("questions", []))
    print(f"Total questions: {total_questions}\n")
    
    def ask(question):
        resp = requests.post(f"{COORDINATOR_URL}/ask", params={"question": question})
        return resp.json().get("answer")
    
    for idx, question in enumerate(questions.get("questions", []), start=1):
        print(f"\nQuestion {idx}/{total_questions}: {question.get('question')}")
        output = measure_function(ask, question.get("question"))
        print(f"Answer: {output['result']}")
        
        answer_received = output["result"]
        metrics = output.get("metrics", {})
        
        similarity_string = check_similarity_string(question.get("answer"), answer_received)
        similarity_semantic = check_similarity_semantic(question.get("answer"), answer_received)
        
        print(f"Similarity (string): {similarity_string:.3f}")
        print(f"Similarity (semantic): {similarity_semantic:.3f}")
        print(f"Similarity (average): {(similarity_string + similarity_semantic) / 2:.3f}")
        
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
        
        qa.append(row)
    
    print("\n" + "="*60)
    print(f"COMPLETED Q&A: {len(qa)} QUESTIONS")
    print("="*60 + "\n")
    
    return qa

def export_qa(qa, MEMORY_BACKEND, conversation_index):
    if not qa:
        print("No QA results to save.")
        return
    
    print(f"Exporting {len(qa)} QA result(s)...")
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
    print(f"Constrained mode: {is_constrained}")
    
    output_file = f"{MEMORY_BACKEND}_conv_{conversation_index}_qa{'_constrained' if is_constrained else ''}.csv"
    output_path = os.path.join(output_dir, output_file)
    
    if os.path.exists(output_path):
        backup_file = f"{MEMORY_BACKEND}_conv_{conversation_index}_qa{'_constrained' if is_constrained else ''}_backup.csv"
        backup_path = os.path.join(output_dir, backup_file)
        shutil.move(output_path, backup_path)
        print(f"Existing file backed up to {backup_path}")
    
    df.to_csv(output_path, index=False)
    print(f"✓ Saved new QA results to {output_path}\n")
