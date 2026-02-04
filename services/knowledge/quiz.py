
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from services.llm import get_llm_client, safe_json_parse
from knowledge.models import KnowledgeNode

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """너는 컴퓨터 과학 및 인지과학 분야의 전문가이자 교육자야.
주어진 지식 개념(Node)에 대해 학습자의 이해도를 평가할 수 있는 4지 선다형 퀴즈를 만들어줘.

## 규칙
1. 문제는 해당 개념의 핵심 내용이나 응용 사례를 묻는 것이어야 해.
2. 보기는 총 4개여야 하며, 정답은 하나뿐이어야 해.
3. 정답은 무작위 위치에 배치해 (항상 1번이거나 하지 않도록).
4. 오답은 매력적이어야 하며(단순 오타나 말장난 금지), 개념을 잘못 이해했을 때 고를 법한 내용이어야 해.
5. 해설(explanation)은 정답인 이유와 오답인 이유를 명확히 설명해줘.
6. 한국어로 작성해줘.
7. 출력은 반드시 JSON 형식이어야 해.

## JSON 출력 예시
{
  "question": "Python의 리스트(List)와 튜플(Tuple)의 가장 큰 차이점은 무엇인가?",
  "options": [
    "리스트는 순서가 있지만 튜플은 순서가 없다.",
    "리스트는 변경 가능(Mutable)하지만 튜플은 변경 불가능(Immutable)하다.",
    "리스트는 숫자만 저장할 수 있고 튜플은 문자열만 저장할 수 있다.",
    "튜플이 리스트보다 메모리를 더 많이 사용한다."
  ],
  "answer_index": 1,
  "explanation": "리스트는 생성 후 요소를 수정할 수 있는 Mutable 객체이지만, 튜플은 한 번 생성되면 수정할 수 없는 Immutable 객체입니다."
}
"""

from pydantic import BaseModel, Field
from typing import List

class QuizSchema(BaseModel):
    question: str
    options: List[str]
    answer_index: int
    explanation: str = ""

class QuizGenerator:
    """지식 노드 기반 퀴즈 생성기"""
    
    def __init__(self):
        self.client = get_llm_client()

    def generate_multiple_choice(self, node: KnowledgeNode) -> Dict[str, Any]:
        """
        특정 노드에 대한 4지 선다 퀴즈 생성
        
        Args:
            node: KnowledgeNode 인스턴스
            
        Returns:
            Dict: {question, options, answer_index, explanation}
        """
        user_prompt = f"""
다음 개념에 대한 4지 선다 퀴즈를 1개 만들어줘:

- 개념 제목: {node.title}
- 개념 설명: {node.description}
- 태그: {', '.join(node.tags)}
"""
        try:
            # LLM 호출
            response_text = self.client.generate_with_system(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            
            # JSON 파싱 (Pydantic 모델 사용)
            quiz_model = safe_json_parse(response_text, QuizSchema)
            
            if not quiz_model:
                logger.error("퀴즈 생성 실패: 파싱 결과 없음")
                raise ValueError("LLM 응답을 파싱할 수 없습니다.")
            
            # Pydantic 모델 -> Dict 변환
            quiz_data = quiz_model.model_dump()
            
            return quiz_data
            
        except Exception as e:
            logger.error(f"퀴즈 생성 중 오류 발생 ({node.title}): {e}")
            raise e
