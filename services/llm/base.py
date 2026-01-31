"""
LLM Client 추상 베이스 클래스

이 모듈은 모든 LLM 클라이언트가 구현해야 하는 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """LLM 클라이언트 설정"""
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 30.0
    
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("API key는 필수입니다.")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature는 0~2 사이여야 합니다.")
        if self.max_tokens < 1:
            raise ValueError("max_tokens는 1 이상이어야 합니다.")


class LLMClient(ABC):
    """
    LLM 클라이언트 추상 베이스 클래스
    
    모든 LLM 구현체는 이 클래스를 상속하고
    generate 메서드를 구현해야 합니다.
    
    Example:
        class MyLLMClient(LLMClient):
            def generate(self, prompt: str) -> str:
                return "response"
    """
    
    def __init__(self, config: LLMConfig):
        """
        Args:
            config: LLM 클라이언트 설정
        """
        self._config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """설정 유효성 검사. 서브클래스에서 오버라이드 가능."""
        pass
    
    @property
    def model(self) -> str:
        """현재 사용 중인 모델 이름"""
        return self._config.model
    
    @property
    def provider(self) -> str:
        """LLM 제공자 이름 (예: 'openai', 'anthropic')"""
        return self.__class__.__name__.lower().replace("client", "")
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        프롬프트를 받아 LLM 응답을 생성합니다.
        
        Args:
            prompt: LLM에 전달할 프롬프트 텍스트
            
        Returns:
            LLM이 생성한 응답 문자열
            
        Raises:
            LLMConnectionError: API 연결 실패 시
            LLMRateLimitError: Rate limit 초과 시
            LLMResponseError: 응답 처리 실패 시
        """
        pass
    
    @abstractmethod
    def generate_with_system(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> str:
        """
        시스템 프롬프트와 사용자 프롬프트를 분리하여 요청합니다.
        
        Args:
            system_prompt: 시스템 레벨 지시사항
            user_prompt: 사용자 입력
            
        Returns:
            LLM이 생성한 응답 문자열
        """
        pass
    
    def generate_json(self, prompt: str) -> str:
        """
        JSON 응답을 요청합니다.
        기본 구현은 generate()를 호출하지만,
        서브클래스에서 JSON 모드를 지원하도록 오버라이드할 수 있습니다.
        
        Args:
            prompt: JSON 형식 응답을 요청하는 프롬프트
            
        Returns:
            JSON 형식의 응답 문자열
        """
        return self.generate(prompt)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"


# =============================================================================
# Custom Exceptions
# =============================================================================

class LLMError(Exception):
    """LLM 관련 기본 예외"""
    pass


class LLMConnectionError(LLMError):
    """API 연결 실패"""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit 초과"""
    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded. Retry after {retry_after}s" 
            if retry_after else "Rate limit exceeded"
        )


class LLMResponseError(LLMError):
    """응답 처리 실패"""
    pass


class LLMAuthenticationError(LLMError):
    """인증 실패"""
    pass
