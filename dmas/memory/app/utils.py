import logging
from typing import Optional
from datetime import datetime
from openai import OpenAI

logger = logging.getLogger(__name__)
client = OpenAI()

@staticmethod
def norm_str(value: Optional[str]) -> str:
    return (value or "").strip()

@staticmethod
def parse_timestamp(ts: Optional[str]) -> Optional[datetime]:  # Changed return type
    if not ts:
        return None
    
    ts_norm = ts.strip().lower().replace("  ", " ")
    
    formats = [
        "%I:%M %p on %d %B, %Y",   # 1:56 pm on 8 May, 2023
        "%I:%M %p on %d %b, %Y",   # 1:56 pm on 8 May, 2023 (short month)
        "%I:%M %p on %d %B %Y",    # 1:56 pm on 8 May 2023 (no comma)
        "%I:%M %p %d %B %Y",       # 1:56 pm 8 May 2023
        "%Y-%m-%dT%H:%M:%S%z",     # ISO-like
        "%Y-%m-%d %H:%M:%S",       # common DB format
        "%Y-%m-%d",                # simple date
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_norm, fmt)
            return dt
        except Exception:
            continue
    
    return None

@staticmethod
def extract_name(question: str) -> Optional[str]:
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the most likely person's name from the user's text. "
                        "Return only the name, nothing else. "
                        "If there is no clear name, return 'none'."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        name = completion.choices[0].message.content.strip()
        if name.lower() == "none" or not name:
            return None
        return name.lower()
    except Exception as e:
        return None