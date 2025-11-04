import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from app.utils import norm_str, extract_name, parse_timestamp

logger = logging.getLogger(__name__)

class GraphitiService:
    def __init__(self):
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        if not (neo4j_uri and neo4j_user and neo4j_password):
            raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")
        
        self.graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
        self._initialized = False
        
    async def _initialize(self):        
        if not self._initialized:
            try:
                logger.info("Starting build_indices_and_constraints...")
                await self.graphiti.build_indices_and_constraints()
                self._initialized = True
                logger.info("Graphiti indices initialized successfully")
            except Exception as e:
                logger.exception("Failed to initialize Graphiti indices: %s", e)
        else:
            logger.info("Already initialized, skipping")
    
    async def memorize_conversation_async(self, conv_index: int, data: Dict[str, Any]) -> Dict[str, Any]:
        await self._initialize()
        
        sessions = data.get("sessions") or {}
        session_datetimes = data.get("session_datetimes") or {}
        
        added = 0
        skipped = 0
        failed = 0
        failures: List[Dict[str, Any]] = []
        episode_results: List[Dict[str, Any]] = []
        
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
            total_turns = len(turns)
            
            logger.info("Processing session %d/%d: %s with %d turns, timestamp=%s", 
                    session_idx, total_sessions, session_key, total_turns, timestamp)
            
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
                
                reference_time = timestamp if timestamp else datetime.now()
                
                episode_content = f"[{speaker} at {reference_time.isoformat()}]: {text}"
                if blip_caption:
                    episode_content += f" (Image: {blip_caption})"
                
                episode_name = f"conversation_{conv_index}_{session_key}_{turn_idx - 1}"
                
                try:
                    logger.info("Session %s [%d/%d]: Turn %d/%d - Adding episode for speaker '%s'", 
                            session_key, session_idx, total_sessions, 
                            turn_idx, total_turns, speaker or "unknown")
                    
                    result = await self.graphiti.add_episode(
                        name=episode_name,
                        episode_body=episode_content,
                        source=EpisodeType.text,
                        source_description=f"Speaker: {speaker}",
                        reference_time=reference_time,
                    )
                    
                    episode_results.append({
                        "session": session_key,
                        "session_index": session_idx,
                        "turn_index": turn_idx - 1,
                        "text_snippet": text[:100],
                        "speaker": speaker,
                        "timestamp": reference_time.isoformat(),
                        "episode_name": episode_name,
                        "episode_result": result,
                    })
                    
                    added += 1
                    logger.info("Session %s [%d/%d]: Turn %d/%d - SUCCESS: Added (total: %d)", 
                            session_key, session_idx, total_sessions, 
                            turn_idx, total_turns, added)
                    
                except Exception as exc:
                    logger.exception("Session %s [%d/%d]: Turn %d/%d - FAILED to add episode", 
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
                    sum(1 for r in episode_results if r["session"] == session_key),
                    sum(1 for i in range(turn_idx) if i not in [r["turn_index"] for r in episode_results if r["session"] == session_key]))
        
        summary = {
            "status": "success" if failed == 0 else "partial_failure",
            "conversation_id": conv_index,
            "added": added,
            "skipped": skipped,
            "failed": failed,
            "total_sessions": total_sessions,
            "total_processed": added + skipped + failed,
            "failures": failures if failures else None,
            "results": episode_results,
        }
        
        return summary
    
    def memorize_conversation(self, conv_index: int, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.memorize_conversation_async(conv_index, data))
    
    async def remember_async(self, question: str) -> List[str]:
        await self._initialize()
        
        try:
            config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            
            results = await self.graphiti._search(query=question, config=config)
            
            memories: List[str] = []
            for node in results.nodes:
                summary = (node.summary or "").strip()
                if summary:
                    memories.append(summary)
            
            return memories
            
        except Exception as e:
            logger.exception("Graphiti search failed: %s", e)
            return []
    
    def remember(self, question: str) -> List[str]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.remember_async(question))