"""
Safe JSON Parser

LLM 응답에서 마크다운 코드 블록을 제거하고,
순수 JSON을 추출하여 Pydantic 모델로 검증합니다.

특징:
- ```json ... ``` 마크다운 블록 자동 제거
- 깨진 JSON 복구 시도
- Pydantic 검증 실패 시 상세 에러 메시지
- 빈 리스트 반환 옵션
"""

import json
import re
import logging
from typing import TypeVar, Type, Optional, Union, List

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class JSONParseError(Exception):
    """JSON 파싱 실패 예외"""
    def __init__(self, message: str, raw_content: str, original_error: Optional[Exception] = None):
        self.raw_content = raw_content
        self.original_error = original_error
        super().__init__(message)


def extract_json_from_markdown(text: str) -> str:
    """
    마크다운 코드 블록에서 JSON을 추출합니다.
    
    지원하는 형식:
    - ```json ... ```
    - ``` ... ```
    - 코드 블록 없는 순수 JSON
    
    Args:
        text: LLM 응답 텍스트
        
    Returns:
        추출된 JSON 문자열
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # 패턴 1: ```json ... ``` 또는 ```JSON ... ```
    json_block_pattern = r"```(?:json|JSON)?\s*\n?([\s\S]*?)\n?```"
    matches = re.findall(json_block_pattern, text)
    
    if matches:
        # 가장 큰 매치 반환 (여러 블록이 있을 경우)
        extracted = max(matches, key=len).strip()
        logger.debug(f"마크다운 블록에서 JSON 추출: {len(extracted)} chars")
        return extracted
    
    # 패턴 2: JSON 객체/배열 직접 찾기
    # { 또는 [로 시작하는 부분 찾기
    json_start_pattern = r"(\{[\s\S]*\}|\[[\s\S]*\])"
    match = re.search(json_start_pattern, text)
    
    if match:
        return match.group(1).strip()
    
    # 그대로 반환
    return text


def repair_json(text: str) -> str:
    """
    깨진 JSON 복구를 시도합니다.
    
    복구 시도:
    - 후행 쉼표 제거
    - 잘린 JSON 닫기
    - 작은따옴표를 큰따옴표로 변환
    """
    if not text:
        return text
    
    # 작은따옴표를 큰따옴표로 (문자열 내부가 아닌 경우만)
    # 주의: 단순 변환은 위험할 수 있음
    
    # 후행 쉼표 제거: ,] 또는 ,}
    text = re.sub(r",\s*([}\]])", r"\1", text)
    
    # 중괄호/대괄호 균형 맞추기
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    
    if open_braces > 0:
        text += "}" * open_braces
        logger.debug(f"닫는 중괄호 {open_braces}개 추가")
    
    if open_brackets > 0:
        text += "]" * open_brackets
        logger.debug(f"닫는 대괄호 {open_brackets}개 추가")
    
    return text


def safe_json_parse(
    content: str,
    model: Type[T],
    *,
    raise_on_error: bool = False,
    return_empty_on_error: bool = True,
    repair: bool = True,
) -> Union[T, List, None]:
    """
    LLM 응답을 안전하게 JSON으로 파싱하고 Pydantic 모델로 검증합니다.
    
    Args:
        content: LLM 응답 문자열 (마크다운 코드 블록 포함 가능)
        model: 검증에 사용할 Pydantic 모델 클래스
        raise_on_error: True이면 실패 시 예외 발생
        return_empty_on_error: True이면 실패 시 빈 리스트/모델 반환
        repair: True이면 깨진 JSON 복구 시도
        
    Returns:
        성공 시: Pydantic 모델 인스턴스
        실패 시: 
            - raise_on_error=True: ValidationError 또는 JSONParseError 발생
            - return_empty_on_error=True: 빈 리스트 [] 또는 빈 모델
            - 둘 다 False: None
            
    Example:
        from services.llm.schemas import NodeListResponse
        
        response = '''```json
        {"nodes": [{"id": "1", "title": "ML", "description": ""}]}
        ```'''
        
        result = safe_json_parse(response, NodeListResponse)
        print(result.nodes)  # [NodeSchema(id='1', title='ML', ...)]
    """
    if not content:
        logger.warning("빈 content가 전달됨")
        return _handle_error(
            "빈 content",
            content or "",
            None,
            model,
            raise_on_error,
            return_empty_on_error,
        )
    
    # Step 1: 마크다운 코드 블록 제거
    json_str = extract_json_from_markdown(content)
    
    if not json_str:
        logger.warning("JSON 추출 실패")
        return _handle_error(
            "JSON을 찾을 수 없습니다",
            content,
            None,
            model,
            raise_on_error,
            return_empty_on_error,
        )
    
    # Step 2: JSON 복구 시도
    if repair:
        json_str = repair_json(json_str)
    
    # Step 3: JSON 파싱
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 파싱 실패: {e}")
        return _handle_error(
            f"JSON 파싱 실패: {e}",
            content,
            e,
            model,
            raise_on_error,
            return_empty_on_error,
        )
    
    # Step 4: Pydantic 검증
    try:
        # 데이터가 리스트인 경우 처리
        if isinstance(data, list):
            # 모델이 리스트를 items 또는 nodes 등으로 감싸는지 확인
            model_fields = model.model_fields
            list_field = None
            
            for field_name, field_info in model_fields.items():
                # List 타입 필드 찾기
                origin = getattr(field_info.annotation, "__origin__", None)
                if origin is list:
                    list_field = field_name
                    break
            
            if list_field:
                data = {list_field: data}
            else:
                # 단일 항목 리스트
                if len(data) == 1:
                    data = data[0]
        
        result = model.model_validate(data)
        logger.debug(f"Pydantic 검증 성공: {model.__name__}")
        return result
        
    except ValidationError as e:
        logger.warning(f"Pydantic 검증 실패: {e}")
        return _handle_error(
            f"데이터 검증 실패: {e}",
            content,
            e,
            model,
            raise_on_error,
            return_empty_on_error,
        )


def _handle_error(
    message: str,
    raw_content: str,
    original_error: Optional[Exception],
    model: Type[T],
    raise_on_error: bool,
    return_empty_on_error: bool,
) -> Union[T, List, None]:
    """에러 처리 헬퍼"""
    if raise_on_error:
        if isinstance(original_error, ValidationError):
            raise original_error
        raise JSONParseError(message, raw_content, original_error)
    
    if return_empty_on_error:
        # 빈 모델 생성 시도
        try:
            return model.model_validate({})
        except ValidationError:
            # 필수 필드가 있으면 빈 리스트 반환
            return []
    
    return None


def safe_json_parse_list(
    content: str,
    item_model: Type[T],
    *,
    raise_on_error: bool = False,
) -> List[T]:
    """
    JSON 배열을 파싱하여 Pydantic 모델 리스트로 반환합니다.
    
    Args:
        content: JSON 배열이 포함된 LLM 응답
        item_model: 배열 항목의 Pydantic 모델
        raise_on_error: True이면 실패 시 예외 발생
        
    Returns:
        Pydantic 모델 인스턴스 리스트 (실패 시 빈 리스트)
    """
    json_str = extract_json_from_markdown(content)
    json_str = repair_json(json_str)
    
    try:
        data = json.loads(json_str)
        
        if not isinstance(data, list):
            # 단일 객체면 리스트로 감싸기
            data = [data]
        
        results = []
        for item in data:
            try:
                results.append(item_model.model_validate(item))
            except ValidationError as e:
                logger.warning(f"항목 검증 실패, 스킵: {e}")
                if raise_on_error:
                    raise
        
        return results
        
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"리스트 파싱 실패: {e}")
        if raise_on_error:
            raise
        return []
