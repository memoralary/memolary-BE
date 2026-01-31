"""
Anthropic Claude LLM Client 구현
"""

import logging
from typing import Optional

from services.llm.base import (
    LLMClient,
    LLMConfig,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMAuthenticationError,
)

logger = logging.getLogger(__name__)


class ClaudeClient(LLMClient):
    """
    Anthropic Claude API 클라이언트 구현
    
    Supported models:
        - claude-sonnet-4-20250514 (default)
        - claude-3-5-sonnet-20241022
        - claude-3-5-haiku-20241022
        - claude-3-opus-20240229
    
    Example:
        config = LLMConfig(
            api_key="sk-ant-...",
            model="claude-sonnet-4-20250514",
            temperature=0.7
        )
        client = ClaudeClient(config)
        response = client.generate("Hello!")
    """
    
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    SUPPORTED_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Optional["anthropic.Anthropic"] = None
    
    def _validate_config(self) -> None:
        """Claude 설정 유효성 검사"""
        if not self._config.api_key.startswith("sk-ant-"):
            logger.warning("Anthropic API key가 올바른 형식이 아닐 수 있습니다.")
    
    @property
    def client(self) -> "anthropic.Anthropic":
        """Lazy initialization of Anthropic client"""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self._config.api_key,
                    timeout=self._config.timeout,
                )
            except ImportError:
                raise ImportError(
                    "anthropic 패키지가 설치되지 않았습니다. "
                    "'pip install anthropic'를 실행하세요."
                )
        return self._client
    
    def generate(self, prompt: str) -> str:
        """단일 프롬프트로 응답 생성"""
        return self.generate_with_system("", prompt)
    
    def generate_with_system(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> str:
        """시스템/사용자 프롬프트로 응답 생성"""
        return self._call_api(
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
    
    def generate_json(self, prompt: str) -> str:
        """
        JSON 응답 생성
        
        Note: Claude는 네이티브 JSON 모드가 없으므로
        프롬프트에 JSON 출력을 명시하는 것이 좋습니다.
        """
        json_prompt = (
            f"{prompt}\n\n"
            "반드시 유효한 JSON 형식으로만 응답하세요. "
            "다른 텍스트 없이 JSON만 출력하세요."
        )
        return self.generate(json_prompt)
    
    def _call_api(
        self, 
        messages: list,
        system: str = ""
    ) -> str:
        """Anthropic API 호출"""
        import anthropic
        
        try:
            kwargs = {
                "model": self._config.model,
                "messages": messages,
                "max_tokens": self._config.max_tokens,
            }
            
            # temperature는 0보다 커야 함 (Claude 제약)
            if self._config.temperature > 0:
                kwargs["temperature"] = self._config.temperature
            
            # 시스템 프롬프트가 있으면 추가
            if system:
                kwargs["system"] = system
            
            response = self.client.messages.create(**kwargs)
            
            # 응답 추출
            if not response.content:
                raise LLMResponseError("Claude 응답에 content가 없습니다.")
            
            content = response.content[0].text
            
            logger.debug(
                f"Claude 응답 생성 완료: "
                f"model={self._config.model}, "
                f"input_tokens={response.usage.input_tokens}, "
                f"output_tokens={response.usage.output_tokens}"
            )
            
            return content
            
        except anthropic.AuthenticationError as e:
            logger.error(f"Claude 인증 실패: {e}")
            raise LLMAuthenticationError(str(e))
            
        except anthropic.RateLimitError as e:
            logger.warning(f"Claude Rate limit 초과: {e}")
            raise LLMRateLimitError()
            
        except anthropic.APIConnectionError as e:
            logger.error(f"Claude 연결 실패: {e}")
            raise LLMConnectionError(str(e))
            
        except anthropic.APIError as e:
            logger.error(f"Claude API 에러: {e}")
            raise LLMResponseError(str(e))
