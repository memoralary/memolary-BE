"""
LLM Client Factory

환경 변수를 기반으로 적절한 LLM 클라이언트를 생성합니다.

환경 변수:
    LLM_PROVIDER: 사용할 LLM 제공자 ('openai' 또는 'claude')
    OPENAI_API_KEY: OpenAI API 키
    ANTHROPIC_API_KEY: Anthropic API 키
    LLM_MODEL: 사용할 모델명 (선택적)
    LLM_TEMPERATURE: 생성 temperature (선택적, 기본값: 0.7)
    LLM_MAX_TOKENS: 최대 토큰 수 (선택적, 기본값: 4096)
"""

import os
import logging
from enum import Enum
from functools import lru_cache
from typing import Optional

from services.llm.base import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """지원되는 LLM 제공자"""
    OPENAI = "openai"
    CLAUDE = "claude"
    ANTHROPIC = "anthropic"  # claude의 별칭


def get_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_cache: bool = True,
) -> LLMClient:
    """
    환경 변수 또는 인자를 기반으로 LLM 클라이언트를 생성합니다.
    
    Args:
        provider: LLM 제공자 ('openai' 또는 'claude'). 
                  None이면 환경변수 LLM_PROVIDER 사용
        api_key: API 키. None이면 환경변수에서 자동 로드
        model: 모델명. None이면 제공자의 기본 모델 사용
        temperature: 생성 temperature (0~2)
        max_tokens: 최대 토큰 수
        use_cache: True이면 동일 설정에 대해 캐시된 클라이언트 반환
        
    Returns:
        LLMClient: 설정된 LLM 클라이언트 인스턴스
        
    Raises:
        ValueError: 지원되지 않는 provider거나 API 키가 없을 때
        
    Example:
        # 환경변수 기반 (권장)
        client = get_llm_client()
        
        # 명시적 설정
        client = get_llm_client(
            provider="openai",
            model="gpt-4o",
            temperature=0.5
        )
    """
    # Provider 결정
    resolved_provider = (
        provider or 
        os.getenv("LLM_PROVIDER", "openai")
    ).lower()
    
    # claude와 anthropic을 동일하게 처리
    if resolved_provider == "anthropic":
        resolved_provider = "claude"
    
    # API 키 결정
    if api_key is None:
        if resolved_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif resolved_provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError(
            f"{resolved_provider.upper()} API 키가 필요합니다. "
            f"환경변수 또는 api_key 인자로 전달하세요."
        )
    
    # 기타 설정
    resolved_model = model or os.getenv("LLM_MODEL")
    resolved_temperature = temperature or float(os.getenv("LLM_TEMPERATURE", "0.7"))
    resolved_max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", "4096"))
    
    # 캐시 사용 시 캐시된 클라이언트 반환
    if use_cache:
        return _get_cached_client(
            resolved_provider,
            api_key,
            resolved_model,
            resolved_temperature,
            resolved_max_tokens,
        )
    
    return _create_client(
        resolved_provider,
        api_key,
        resolved_model,
        resolved_temperature,
        resolved_max_tokens,
    )


@lru_cache(maxsize=4)
def _get_cached_client(
    provider: str,
    api_key: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> LLMClient:
    """캐시된 클라이언트 반환"""
    return _create_client(provider, api_key, model, temperature, max_tokens)


def _create_client(
    provider: str,
    api_key: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> LLMClient:
    """실제 클라이언트 생성"""
    if provider == "openai":
        from services.llm.openai_client import OpenAIClient
        
        config = LLMConfig(
            api_key=api_key,
            model=model or OpenAIClient.DEFAULT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        client = OpenAIClient(config)
        
    elif provider == "claude":
        from services.llm.claude_client import ClaudeClient
        
        config = LLMConfig(
            api_key=api_key,
            model=model or ClaudeClient.DEFAULT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        client = ClaudeClient(config)
        
    else:
        raise ValueError(
            f"지원되지 않는 LLM provider: {provider}. "
            f"'openai' 또는 'claude'를 사용하세요."
        )
    
    logger.info(f"LLM 클라이언트 생성: {client}")
    return client


def clear_client_cache() -> None:
    """클라이언트 캐시 초기화"""
    _get_cached_client.cache_clear()
    logger.info("LLM 클라이언트 캐시 초기화됨")
