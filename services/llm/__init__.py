"""
LLM Service Package

사용법:
    from services.llm import get_llm_client, safe_json_parse
    
    client = get_llm_client()  # 환경변수에 따라 자동 선택
    response = client.generate("Hello, world!")
    
    # JSON 파싱
    from services.llm.schemas import NodeListResponse
    result = safe_json_parse(response, NodeListResponse)
"""

from services.llm.factory import get_llm_client
from services.llm.parser import safe_json_parse

__all__ = [
    "get_llm_client",
    "safe_json_parse",
]
