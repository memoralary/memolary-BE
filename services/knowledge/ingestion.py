"""
Ingestion Service - 멀티모달 입력 처리

텍스트 또는 PDF 파일을 통합된 문자열/청크로 변환합니다.

사용법:
    from services.knowledge.ingestion import IngestionService
    
    service = IngestionService()
    
    # 텍스트 입력
    chunks = service.process("머신러닝은 데이터로부터...")
    
    # PDF 파일 입력
    chunks = service.process("/path/to/document.pdf")
    
    # extract_nodes와 연계
    for chunk in chunks:
        result = extract_nodes(chunk)
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Union, Optional, BinaryIO
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class IngestionError(Exception):
    """Ingestion 관련 기본 예외"""
    pass


class PDFExtractionError(IngestionError):
    """PDF 텍스트 추출 실패"""
    pass


class OCRRequiredError(IngestionError):
    """OCR이 필요한 이미지 기반 PDF"""
    pass


class UnsupportedFormatError(IngestionError):
    """지원하지 않는 파일 형식"""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IngestionResult:
    """Ingestion 결과"""
    chunks: List[str]               # 청크 리스트
    source_type: str                # "text", "pdf", "file"
    source_name: str                # 소스 이름 (파일명 또는 "direct_input")
    total_chars: int                # 총 문자 수
    page_count: int = 0             # PDF 페이지 수
    warnings: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """전체 텍스트 반환 (청크 결합)"""
        return "\n\n".join(self.chunks)
    
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


# =============================================================================
# Text Cleaner
# =============================================================================

class TextCleaner:
    """텍스트 정제 유틸리티"""
    
    @staticmethod
    def clean(text: str) -> str:
        """기본 텍스트 정제"""
        if not text:
            return ""
        
        # 연속 공백 제거
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 3개 이상 연속 줄바꿈 → 2개로
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 줄 앞뒤 공백 제거
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def remove_headers_footers(text: str) -> str:
        """페이지 번호, 헤더/푸터 제거"""
        # 독립된 숫자만 있는 줄 제거 (페이지 번호)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # 숫자만 있는 줄 제거
            if stripped and stripped.isdigit():
                continue
            # "Page X", "- X -" 패턴 제거
            if re.match(r'^(Page|페이지)?\s*\d+\s*$', stripped, re.IGNORECASE):
                continue
            if re.match(r'^[-–—]\s*\d+\s*[-–—]$', stripped):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


# =============================================================================
# PDF Extractor
# =============================================================================

class PDFExtractor:
    """PyMuPDF 기반 PDF 텍스트 추출"""
    
    # 텍스트가 부족할 때 OCR이 필요하다고 판단하는 임계값
    MIN_CHARS_PER_PAGE = 50
    
    def __init__(self):
        self._fitz = None
    
    @property
    def fitz(self):
        """Lazy import of PyMuPDF"""
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
            except ImportError:
                raise ImportError(
                    "PyMuPDF 패키지가 필요합니다. "
                    "'pip install pymupdf'를 실행하세요."
                )
        return self._fitz
    
    def extract(
        self,
        source: Union[str, Path, BinaryIO],
        clean_text: bool = True
    ) -> IngestionResult:
        """
        PDF에서 텍스트 추출
        
        Args:
            source: 파일 경로 또는 파일 객체
            clean_text: 텍스트 정제 여부
            
        Returns:
            IngestionResult
        """
        warnings = []
        
        # 파일 열기
        if isinstance(source, (str, Path)):
            source_name = Path(source).name
            doc = self.fitz.open(str(source))
        else:
            # 파일 객체
            source_name = getattr(source, 'name', 'uploaded_file.pdf')
            file_bytes = source.read()
            doc = self.fitz.open(stream=file_bytes, filetype="pdf")
        
        try:
            page_texts = []
            low_text_pages = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # OCR 필요 여부 확인
                if len(text.strip()) < self.MIN_CHARS_PER_PAGE:
                    low_text_pages.append(page_num + 1)
                
                if clean_text:
                    text = TextCleaner.clean(text)
                    text = TextCleaner.remove_headers_footers(text)
                
                if text.strip():
                    page_texts.append(text)
            
            # OCR 경고
            if low_text_pages:
                if len(low_text_pages) == len(doc):
                    # 모든 페이지가 이미지 기반
                    warnings.append(
                        f"⚠️ 이미지 기반 PDF로 보입니다. OCR을 사용하세요. "
                        f"(모든 {len(doc)}페이지에서 텍스트가 거의 없음)"
                    )
                else:
                    warnings.append(
                        f"⚠️ 일부 페이지에서 텍스트가 적습니다: {low_text_pages}. "
                        f"해당 페이지는 이미지일 수 있습니다."
                    )
            
            # 텍스트가 전혀 없는 경우
            if not page_texts:
                raise OCRRequiredError(
                    "PDF에서 텍스트를 추출할 수 없습니다. "
                    "이미지 기반 PDF일 수 있습니다. OCR이 필요합니다."
                )
            
            return IngestionResult(
                chunks=page_texts,
                source_type="pdf",
                source_name=source_name,
                total_chars=sum(len(t) for t in page_texts),
                page_count=len(doc),
                warnings=warnings
            )
        
        finally:
            doc.close()


# =============================================================================
# Ingestion Service
# =============================================================================

class IngestionService:
    """
    멀티모달 입력 처리 서비스
    
    텍스트 또는 PDF 파일을 청크 단위로 변환합니다.
    
    Example:
        service = IngestionService(chunk_size=2000)
        
        # 텍스트
        result = service.process("머신러닝은...")
        
        # PDF
        result = service.process("/path/to/document.pdf")
        
        # extract_nodes와 연계
        for chunk in result.chunks:
            nodes = extract_nodes(chunk)
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}
    
    def __init__(
        self,
        chunk_size: int = 4000,  # 속도 최적화: 2000 → 4000 (API 호출 절반)
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """
        Args:
            chunk_size: 청크 최대 문자 수
            chunk_overlap: 청크 간 겹침 문자 수
            min_chunk_size: 최소 청크 크기 (이보다 작으면 이전 청크에 병합)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.pdf_extractor = PDFExtractor()
        self.text_cleaner = TextCleaner()
    
    def process(
        self,
        source: Union[str, Path, BinaryIO],
        *,
        chunk: bool = True,
        clean: bool = True
    ) -> IngestionResult:
        """
        입력을 처리하여 청크로 반환
        
        Args:
            source: 텍스트 문자열, 파일 경로, 또는 파일 객체
            chunk: 청크로 분할할지 여부
            clean: 텍스트 정제 여부
            
        Returns:
            IngestionResult
        """
        # 입력 타입 판별
        if isinstance(source, (Path, BinaryIO)):
            return self._process_file(source, chunk=chunk, clean=clean)
        
        if isinstance(source, str):
            # 파일 경로인지 확인
            if os.path.isfile(source):
                return self._process_file(source, chunk=chunk, clean=clean)
            else:
                # 직접 입력 텍스트
                return self._process_text(source, chunk=chunk, clean=clean)
        
        raise UnsupportedFormatError(f"지원하지 않는 입력 타입: {type(source)}")
    
    def _process_text(
        self,
        text: str,
        *,
        chunk: bool = True,
        clean: bool = True
    ) -> IngestionResult:
        """텍스트 처리"""
        if clean:
            text = self.text_cleaner.clean(text)
        
        if chunk:
            chunks = self._split_into_chunks(text)
        else:
            chunks = [text] if text.strip() else []
        
        return IngestionResult(
            chunks=chunks,
            source_type="text",
            source_name="direct_input",
            total_chars=len(text),
            page_count=0
        )
    
    def _process_file(
        self,
        source: Union[str, Path, BinaryIO],
        *,
        chunk: bool = True,
        clean: bool = True
    ) -> IngestionResult:
        """파일 처리"""
        # 확장자 확인
        if isinstance(source, (str, Path)):
            ext = Path(source).suffix.lower()
            source_name = Path(source).name
        else:
            # 파일 객체
            source_name = getattr(source, 'name', 'uploaded_file')
            ext = Path(source_name).suffix.lower()
        
        if ext == '.pdf':
            result = self.pdf_extractor.extract(source, clean_text=clean)
            
            if chunk:
                # 페이지별 청크를 추가로 분할
                all_chunks = []
                for page_text in result.chunks:
                    all_chunks.extend(self._split_into_chunks(page_text))
                result.chunks = all_chunks
            
            return result
        
        elif ext in {'.txt', '.md'}:
            # 텍스트 파일
            if isinstance(source, (str, Path)):
                with open(source, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = source.read()
                if isinstance(text, bytes):
                    text = text.decode('utf-8')
            
            if clean:
                text = self.text_cleaner.clean(text)
            
            if chunk:
                chunks = self._split_into_chunks(text)
            else:
                chunks = [text] if text.strip() else []
            
            return IngestionResult(
                chunks=chunks,
                source_type="file",
                source_name=source_name,
                total_chars=len(text),
                page_count=0
            )
        
        else:
            raise UnsupportedFormatError(
                f"지원하지 않는 파일 형식: {ext}. "
                f"지원 형식: {self.SUPPORTED_EXTENSIONS}"
            )
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할
        
        단락(빈 줄) 기준으로 분할하고, 최대 크기를 초과하면 문장 단위로 분할
        """
        if not text or not text.strip():
            return []
        
        # 단락 기준 분할
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 현재 청크 + 새 단락이 최대 크기 이내
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 단락 자체가 최대 크기 초과시 문장 단위로 분할
                if len(para) > self.chunk_size:
                    sentences = self._split_by_sentence(para)
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                            if current_chunk:
                                current_chunk += " " + sent
                            else:
                                current_chunk = sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent
                else:
                    current_chunk = para
        
        # 마지막 청크
        if current_chunk:
            # 최소 크기 미달이면 이전 청크에 병합
            if len(current_chunk) < self.min_chunk_size and chunks:
                chunks[-1] += "\n\n" + current_chunk
            else:
                chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_sentence(self, text: str) -> List[str]:
        """문장 단위 분할"""
        # 한국어/영어 문장 구분자
        sentences = re.split(r'(?<=[.!?。])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# =============================================================================
# Convenience Functions
# =============================================================================

def ingest(
    source: Union[str, Path, BinaryIO],
    chunk_size: int = 2000
) -> List[str]:
    """
    간편한 ingestion 함수
    
    Args:
        source: 텍스트, 파일 경로, 또는 파일 객체
        chunk_size: 청크 최대 크기
        
    Returns:
        청크 리스트
    """
    service = IngestionService(chunk_size=chunk_size)
    result = service.process(source)
    
    # 경고 출력
    for warning in result.warnings:
        logger.warning(warning)
    
    return result.chunks


def ingest_text(text: str, chunk_size: int = 2000) -> List[str]:
    """텍스트를 청크로 분할"""
    return ingest(text, chunk_size=chunk_size)


def ingest_pdf(pdf_path: str, chunk_size: int = 2000) -> List[str]:
    """PDF를 청크로 분할"""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {pdf_path}")
    return ingest(pdf_path, chunk_size=chunk_size)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 IngestionService 테스트")
    print("=" * 60)
    
    service = IngestionService(chunk_size=500)
    
    # 텍스트 테스트
    sample_text = """
    머신러닝(Machine Learning)은 인공지능의 한 분야로,
    데이터로부터 패턴을 학습하여 예측하는 알고리즘이다.
    
    지도 학습은 정답이 있는 데이터로 학습한다.
    비지도 학습은 정답 없이 패턴을 발견한다.
    
    딥러닝은 심층 신경망을 사용한다.
    """
    
    result = service.process(sample_text)
    
    print(f"\n📊 결과:")
    print(f"   소스: {result.source_type}")
    print(f"   청크 수: {result.chunk_count}")
    print(f"   총 문자: {result.total_chars}")
    
    print(f"\n📦 청크:")
    for i, chunk in enumerate(result.chunks, 1):
        print(f"   [{i}] {chunk[:50]}...")
    
    print("\n🎉 테스트 완료!")
