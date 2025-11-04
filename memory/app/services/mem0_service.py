import logging
import os
from typing import Dict, Any, Optional, List

from mem0 import Memory
from openai import OpenAI
from app.utils import norm_str, extract_name, parse_timestamp

logger = logging.getLogger(__name__)

openai_client = OpenAI()

class Mem0Service:
        
    def __init__(self):
        self.memory = Memory()
    
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
                
                line = f"{timestamp_str}: {text}" if timestamp_str else text
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
        
        name = extract_name(question)
        if name:
            name = name.lower()
        logger.info("Extracted name: '%s'", name)
                
        search_results = self.memory.search(question, user_id=name)
        
        if isinstance(search_results, dict):
            memories = search_results.get("results", [])
        elif isinstance(search_results, list):
            memories = search_results
        else:
            memories = []
        
        logger.info("Processing %d memories...", len(memories))
        
        results: List[str] = []
        for idx, result in enumerate(memories):
            if isinstance(result, dict):
                memory_text = (result.get("memory") or "").strip()
                if memory_text:
                    results.append(memory_text)
            elif isinstance(result, str):
                results.append(result.strip())
        
        return results