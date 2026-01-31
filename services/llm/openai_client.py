"""
OpenAI LLM Client 구현
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


class OpenAIClient(LLMClient):
    """
    OpenAI API 클라이언트 구현
    
    Supported models:
        - gpt-4o (default)
        - gpt-4o-mini
        - gpt-4-turbo
        - gpt-3.5-turbo
    
    Example:
        config = LLMConfig(
            api_key="sk-...",
            model="gpt-4o",
            temperature=0.7
        )
        client = OpenAIClient(config)
        response = client.generate("Hello!")
    """
    
    DEFAULT_MODEL = "gpt-4o"
    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
    ]
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Optional["openai.OpenAI"] = None
    
    def _validate_config(self) -> None:
        """OpenAI 설정 유효성 검사"""
        if not self._config.api_key.startswith("sk-"):
            logger.warning("OpenAI API key가 올바른 형식이 아닐 수 있습니다.")
    
    @property
    def client(self) -> "openai.OpenAI":
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self._config.api_key,
                    timeout=self._config.timeout,
                )
            except ImportError:
                raise ImportError(
                    "openai 패키지가 설치되지 않았습니다. "
                    "'pip install openai'를 실행하세요."
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
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        return self._call_api(messages)
    
    def generate_json(self, prompt: str) -> str:
        """JSON 모드로 응답 생성"""
        return self._call_api(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
    
    def _call_api(
        self, 
        messages: list, 
        response_format: Optional[dict] = None
    ) -> str:
        """OpenAI API 호출"""
        import openai
        
        try:
            kwargs = {
                "model": self._config.model,
                "messages": messages,
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens,
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = self.client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            if content is None:
                raise LLMResponseError("OpenAI 응답에 content가 없습니다.")
            
            logger.debug(
                f"OpenAI 응답 생성 완료: "
                f"model={self._config.model}, "
                f"tokens={response.usage.total_tokens if response.usage else 'N/A'}"
            )
            
            return content
            
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI 인증 실패: {e}")
            raise LLMAuthenticationError(str(e))
            
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI Rate limit 초과: {e}")
            raise LLMRateLimitError()
            
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI 연결 실패: {e}")
            raise LLMConnectionError(str(e))
            
        except openai.APIError as e:
            logger.error(f"OpenAI API 에러: {e}")
            raise LLMResponseError(str(e))
