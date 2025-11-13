import logging
import os
from typing import Dict, Any, Optional, List

from mem0 import Memory
from mem0.configs.base import MemoryConfig
from openai import OpenAI
from app.utils import norm_str, extract_name, parse_timestamp

logger = logging.getLogger(__name__)

openai_client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))

class Mem0Service:
        
    def __init__(self):
        # Configure mem0 to use remote Qdrant server instead of local storage
        # If we don't explicitly set host/port, mem0 defaults to local path='/tmp/qdrant'
        # So ends up using: Client type: QdrantLocal
        
        config = MemoryConfig(
            vector_store={
                "provider": "qdrant",
                "config": {
                    "host": os.getenv("QDRANT_HOST", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", "6333")),
                }
            }
        )
        self.memory = Memory(config)
    
    def memorize_conversation(self, conv_index: int, data: Dict[str, Any]) -> Dict[str, Any]:
        sessions = data.get("sessions") or {}
        session_datetimes = data.get("session_datetimes") or {}
        
        added = 0
        skipped = 0
        failed = 0
        failures: List[Dict[str, Any]] = []
        memory_results: List[Dict[str, Any]] = []
        
        if not isinstance(sessions, dict):
            logger.warning("Expected 'sessions' to be a dict; got %s", type(sessions).__name__)
            return {"status": "error", "reason": "'sessions' must be a dict"}
        
        total_sessions = len(sessions)
        logger.info("Starting memorization for conversation %d with %d sessions", 
                    conv_index, total_sessions)
        
        for session_idx, (session_key, turns) in enumerate(sessions.items(), 1):
            if not isinstance(turns, list):
                logger.debug("Skipping session %s: turns is not a list", session_key)
                skipped += 1
                continue
            
            raw_ts = session_datetimes.get(f"{session_key}_date_time")
            timestamp = parse_timestamp(raw_ts)
            timestamp_str = timestamp.isoformat() if timestamp else None
            total_turns = len(turns)
            
            logger.info("Processing session %d/%d: %s with %d turns, timestamp=%s", 
                    session_idx, total_sessions, session_key, total_turns, timestamp_str)
            
            for turn_idx, turn in enumerate(turns, 1):
                if not isinstance(turn, dict):
                    logger.debug("Session %s: Skipping non-dict turn %d/%d", 
                            session_key, turn_idx, total_turns)
                    skipped += 1
                    continue
                
                text = norm_str(turn.get("text"))
                if not text:
                    logger.debug("Session %s: Skipping empty turn %d/%d", 
                            session_key, turn_idx, total_turns)
                    skipped += 1
                    continue
                
                speaker = norm_str(turn.get("speaker")).lower() or None
                blip_caption = norm_str(turn.get("blip_caption"))
                
                # Store natural text without timestamp prefix to preserve conversational language
                # Timestamp is saved separately in metadata for temporal context
                line = text
                if blip_caption:
                    line += f" (Image: {blip_caption})"
                
                metadata = {
                    "conversation_id": conv_index,
                    "session": session_key,
                    "timestamp": timestamp_str,
                    "turn_index": turn_idx - 1,
                    "speaker": speaker,
                }
                
                try:
                    logger.info("Session %s [%d/%d]: Turn %d/%d - Adding memory for speaker '%s'", 
                            session_key, session_idx, total_sessions, 
                            turn_idx, total_turns, speaker or "unknown")
                    
                    result = self.memory.add(line, user_id=speaker, metadata=metadata)
                    
                    memory_results.append({
                        "session": session_key,
                        "session_index": session_idx,
                        "turn_index": turn_idx - 1,
                        "text_snippet": text[:100],
                        "speaker": speaker,
                        "timestamp": timestamp_str,
                        "memory_result": result,
                    })
                    
                    added += 1
                    logger.info("Session %s [%d/%d]: Turn %d/%d - SUCCESS: Added (total: %d)", 
                            session_key, session_idx, total_sessions, 
                            turn_idx, total_turns, added)
                    
                except Exception as exc:
                    logger.exception("Session %s [%d/%d]: Turn %d/%d - FAILED to add memory", 
                                session_key, session_idx, total_sessions, 
                                turn_idx, total_turns)
                    failed += 1
                    failures.append({
                        "session": session_key,
                        "session_index": session_idx,
                        "turn_index": turn_idx - 1,
                        "text_snippet": text[:100],
                        "error": str(exc),
                    })
            
            logger.info("Session %s [%d/%d]: Completed - Added: %d, Skipped: %d in this session", 
                    session_key, session_idx, total_sessions, 
                    sum(1 for r in memory_results if r["session"] == session_key),
                    sum(1 for i in range(turn_idx) if i not in [r["turn_index"] for r in memory_results if r["session"] == session_key]))
        
        summary = {
            "status": "success" if failed == 0 else "partial_failure",
            "conversation_id": conv_index,
            "added": added,
            "skipped": skipped,
            "failed": failed,
            "total_sessions": total_sessions,
            "total_processed": added + skipped + failed,
            "failures": failures if failures else None,
            "results": memory_results,
        }
        
        return summary

    def remember(
        self,
        question: str
    ) -> List[str]:
        
        # Extract the name to know which user_id to search for
        name = extract_name(question)
        if name:
            name = name.lower()
        logger.info("Extracted name from question: '%s'", name)
        
        # General search parameters: same for all questions
        SEARCH_LIMIT = 100     # Get more results from semantic search
        SEARCH_THRESHOLD = 0.2  # Light threshold: filter only obvious noise
        # The strict responder prompt will filter remaining irrelevant info

        def do_search_for_user(user_id: Optional[str]) -> Any:
            """
            Small helper to call Mem0.search uniformly.
            """
            try:
                logger.info("Mem0 search: query=%r user_id=%r limit=%d threshold=%.2f",
                            question, user_id, SEARCH_LIMIT, SEARCH_THRESHOLD)
                return self.memory.search(
                    question,
                    user_id=user_id,
                    limit=SEARCH_LIMIT,
                    threshold=SEARCH_THRESHOLD,
                )
            except Exception as e:
                logger.exception("Failed to search memories for user %s: %s", user_id, e)
                return []

        # --- Main search ---
        if name:
            search_results = do_search_for_user(name)
        else:
            # No name extracted → fallback to a fixed list of speakers
            logger.warning("No name extracted from question '%s'. Using fallback search.", question)
            known_speakers = ["caroline", "melanie"]
            
            combined_results = []
            for speaker in known_speakers:
                result = do_search_for_user(speaker)
                if isinstance(result, dict) and "results" in result:
                    combined_results.extend(result["results"])
                elif isinstance(result, list):
                    combined_results.extend(result)
            
            search_results = {"results": combined_results} if combined_results else []

        # --- Format results ---
        if isinstance(search_results, dict):
            memories = search_results.get("results", [])
        elif isinstance(search_results, list):
            memories = search_results
        else:
            memories = []

        logger.info("Raw memories from Mem0: %d", len(memories))
        
        # --- Formatting in strings for the model ---
        results: List[str] = []
        for idx, result in enumerate(memories):
            if isinstance(result, dict):
                memory_text = (result.get("memory") or "").strip()
                if not memory_text:
                    continue

                metadata = result.get("metadata", {}) or {}
                timestamp = metadata.get("timestamp")

                formatted_memory = memory_text

                # Add timestamp if available
                if timestamp:
                    try:
                        if isinstance(timestamp, str) and timestamp:
                            date_str = timestamp.split("T")[0] if "T" in timestamp else timestamp[:10]
                            formatted_memory = f"[{date_str}] {formatted_memory}"
                    except Exception as e:
                        logger.debug("Could not parse timestamp: %s", e)

                # Prepend name (if extracted) for clarity
                if name:
                    name_capitalized = name.capitalize()
                    formatted_memory = f"[{name_capitalized}] {formatted_memory}"

                results.append(formatted_memory)

            elif isinstance(result, str):
                txt = result.strip()
                if txt:
                    results.append(txt)

        logger.info("Final formatted memories: %d", len(results))
        return results
