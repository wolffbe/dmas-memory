import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional
import logging

from app.models import Conversation, DialogTurn, Question

logger = logging.getLogger(__name__)

class LocomoService:
    
    def __init__(self, data_url: str):
        self.data_url = data_url
        self.data_path = Path("/data/locomo10.json")
        self.conversations = []
        self.conversations_by_id = {}
        self.all_sessions = []
        self.all_questions = []
        self.is_loaded = False
    
    def download_data(self):
        """Download data file from URL if it doesn't exist locally."""
        if self.data_path.exists():
            logger.info(f"Data file already exists at {self.data_path}")
            return True
        
        logger.info(f"Downloading data from {self.data_url}")
        
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.data_url, self.data_path)
            logger.info(f"Downloaded successfully to {self.data_path}")
            return True
            
        except Exception as e:
            logger.info(f"Error downloading data: {e}")
            return False
        
    def load_from_json(self):
        """Load and parse conversation data from JSON file."""
        if not self.data_path.exists():
            if not self.download_data():
                return False
        
        logger.info(f"Loading conversations from {self.data_path}")
        
        if not self.data_path.exists():
            logger.info(f"File not found: {self.data_path}")
            return False
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.info(f"Expected JSON array, got {type(data)}")
                return False
            
            for conv_idx, conv_data in enumerate(data):
                try:                    
                    conversation_data = conv_data.get("conversation", {})
                    
                    speaker_a = conv_data.get("speaker_a") or conversation_data.get("speaker_a", "")
                    speaker_b = conv_data.get("speaker_b") or conversation_data.get("speaker_b", "")
                                        
                    sessions = {}
                    session_datetimes = {}
                    observations = {}
                    session_summaries = {}
                    event_summary = {}
                    
                    for key, value in conversation_data.items():
                        if key.startswith("session_"):
                            if key.endswith("_date_time") or key.endswith("_observation") or key.endswith("_summary"):
                                if key.endswith("_date_time"):
                                    session_datetimes[key] = value
                                elif key.endswith("_observation"):
                                    if isinstance(value, list):
                                        observations[key] = value
                                elif key.endswith("_summary") and not key.startswith("event"):
                                    session_summaries[key] = value
                            elif isinstance(value, list):
                                sessions[key] = [DialogTurn(**turn) for turn in value]
                        
                        elif key.startswith("events_session_"):
                            event_summary[key] = value
                    
                    if "observation" in conv_data:
                        obs_data = conv_data["observation"]
                        if isinstance(obs_data, dict):
                            observations.update(obs_data)
                    
                    if "session_summary" in conv_data:
                        sum_data = conv_data["session_summary"]
                        if isinstance(sum_data, dict):
                            session_summaries.update(sum_data)
                    
                    if "event_summary" in conv_data:
                        ev_data = conv_data["event_summary"]
                        if isinstance(ev_data, dict):
                            event_summary.update(ev_data)
                    
                    qa = None
                    if "qa" in conv_data and conv_data["qa"]:
                        qa = [Question(**q) for q in conv_data["qa"]]
                    
                    conversation = Conversation(
                        sample_id=conv_data.get("sample_id", ""),
                        speaker_a=speaker_a,
                        speaker_b=speaker_b,
                        sessions=sessions,
                        session_datetimes=session_datetimes,
                        observations=observations if observations else None,
                        session_summaries=session_summaries if session_summaries else None,
                        event_summary=event_summary if event_summary else None,
                        qa=qa
                    )
                    
                    self.conversations.append(conversation)
                    self.conversations_by_id[conversation.sample_id] = conversation
                                        
                    for session_key, turns in sessions.items():
                        session_text = " ".join([turn.text for turn in turns])
                        self.all_sessions.append({
                            "sample_id": conversation.sample_id,
                            "session_id": session_key,
                            "text": session_text,
                            "speakers": [conversation.speaker_a, conversation.speaker_b],
                            "num_turns": len(turns),
                            "date_time": session_datetimes.get(f"{session_key}_date_time"),
                            "turns": [turn.dict() for turn in turns]
                        })
                    
                    if qa:
                        for idx, question in enumerate(qa):
                            q_dict = {
                                "sample_id": conversation.sample_id,
                                "question_id": f"{conversation.sample_id}_q_{idx}",
                                "question": question.question,
                                "answer": question.get_answer(),
                                "category": question.category,
                                "evidence": question.evidence
                            }
                            self.all_questions.append(q_dict)
                    
                except Exception as e:
                    logger.info(f"Error parsing conversation: {e}")
                    continue
            
            self.is_loaded = True
            logger.info(f"Loaded {len(self.conversations)} conversations")
            logger.info(f"Total sessions: {len(self.all_sessions)}")
            logger.info(f"Total questions: {len(self.all_questions)}")
            return True
            
        except json.JSONDecodeError as e:
            logger.info(f"JSON parse error: {e}")
            return False
        except Exception as e:
            logger.info(f"Error loading data: {e}")
            return False
    
    def get_conversation(self, sample_id: str) -> Optional[Conversation]:
        """Get conversation by sample_id."""
        return self.conversations_by_id.get(sample_id)
    
    def get_conversation_by_index(self, index: int) -> Optional[Conversation]:
        """Get conversation by index in the list."""
        if 0 <= index < len(self.conversations):
            return self.conversations[index]
        return None
    
    def get_all_conversations(self, skip: int = 0, limit: int = 10) -> List[Conversation]:
        """Get all conversations with pagination."""
        return self.conversations[skip:skip + limit]
    
    def get_conversation_session_by_id(self, sample_id: str, session_id: str) -> Optional[Dict]:
        """Get a specific session by conversation sample_id and session_id."""
        conversation = self.get_conversation(sample_id)
        if not conversation:
            return None
        
        if session_id in conversation.sessions:
            return {
                "session_id": session_id,
                "turns": [turn.dict() for turn in conversation.sessions[session_id]],
                "date_time": conversation.session_datetimes.get(f"{session_id}_date_time"),
                "num_turns": len(conversation.sessions[session_id])
            }
        return None
    
    def get_conversation_session(self, sample_id: str, session_index: int) -> Optional[Dict]:
        """Get a session by conversation sample_id and session index."""
        conversation = self.get_conversation(sample_id)
        if not conversation:
            return None
        
        session_keys = sorted([k for k in conversation.sessions.keys() if k.startswith("session_")])
        
        if 0 <= session_index < len(session_keys):
            session_key = session_keys[session_index]
            return {
                "session_index": session_index,
                "session_id": session_key,
                "turns": [turn.dict() for turn in conversation.sessions[session_key]],
                "date_time": conversation.session_datetimes.get(f"{session_key}_date_time"),
                "num_turns": len(conversation.sessions[session_key])
            }
        return None
    
    def get_conversation_session_by_conv_index(self, conv_index: int, session_index: int) -> Optional[Dict]:
        """Get a session by conversation index and session index."""
        conversation = self.get_conversation_by_index(conv_index)
        if not conversation:
            return None
        
        return self.get_conversation_session(conversation.sample_id, session_index)
    
    def get_conversation_questions(self, sample_id: str) -> List[Dict]:
        """Get all questions for a specific conversation."""
        return [q for q in self.all_questions if q["sample_id"] == sample_id]
    
    def get_conversation_questions_by_index(self, conv_index: int) -> List[Dict]:
        """Get all questions for a conversation by index."""
        conversation = self.get_conversation_by_index(conv_index)
        if not conversation:
            return []
        return self.get_conversation_questions(conversation.sample_id)
    
    def get_question_by_index(self, index: int) -> Optional[Dict]:
        """Get a question by its index in all_questions."""
        if 0 <= index < len(self.all_questions):
            return self.all_questions[index]
        return None
    
    def get_sessions(self, sample_id: Optional[str] = None) -> List[Dict]:
        """Get all sessions, optionally filtered by sample_id."""
        if sample_id:
            return [s for s in self.all_sessions if s["sample_id"] == sample_id]
        return self.all_sessions
    
    def get_questions(self, sample_id: Optional[str] = None, category: Optional[str] = None) -> List[Dict]:
        """Get questions, optionally filtered by sample_id and/or category."""
        questions = self.all_questions
        
        if sample_id:
            questions = [q for q in questions if q["sample_id"] == sample_id]
        
        if category:
            questions = [q for q in questions if q.get("category") == category]
        
        return questions
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded data."""
        total_turns = sum(s["num_turns"] for s in self.all_sessions)
        
        return {
            "total_conversations": len(self.conversations),
            "total_sessions": len(self.all_sessions),
            "total_turns": total_turns,
            "total_questions": len(self.all_questions),
            "conversations_loaded": self.is_loaded,
            "data_file": str(self.data_path)
        }