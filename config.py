"""
AI 의료정보 검증 시스템 설정 관리
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """AI 모델 설정"""
    # 모델 선택 옵션
    USE_OLLAMA: bool = True  # False로 설정하면 커스텀 모델 사용
    USE_CUSTOM_MODEL: bool = False  # True로 설정하면 기존 모델 사용
    
    # Ollama 설정 (5초 목표 초고속 최적화)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"  # 8B → 3B 모델로 3배 속도 향상
    
    # 기존 커스텀 모델 설정
    CUSTOM_MODEL_PATH: str = ""  # 기존 모델 경로를 여기에 입력
    CUSTOM_MODEL_TYPE: str = "huggingface"  # "huggingface", "local", "gguf"
    
    # Colab 스타일 transformers 모델 설정 (GTX 1660 Super 최적화)
    PRIMARY_MODEL_ID: str = "microsoft/DialoGPT-medium"  # 6GB VRAM에 적합한 크기
    FALLBACK_MODELS: List[str] = None
    
    # 의료 특화 모델 옵션들 (선택 가능)
    MEDICAL_MODEL_OPTIONS: List[str] = None
    
    # 임베딩 모델 (768차원으로 데이터베이스와 호환)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # 768차원 다국어 모델
    
    # 생성 파라미터 (GTX 1660 Super 최적화)
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.2
    DO_SAMPLE: bool = True

    # 출력 언어 설정 (하드코딩 방지, 설정/환경변수로 제어)
    FORCE_RESPONSE_LANGUAGE: bool = True
    RESPONSE_LANGUAGE: str = "ko"  # ISO code e.g., 'ko', 'en'
    LANGUAGE_ENFORCEMENT_PROMPT: str = ""  # 커스텀 프롬프트를 환경변수로 주입 가능
    
    # GPU 최적화 설정
    USE_4BIT_QUANTIZATION: bool = True  # 6GB VRAM용 4-bit 양자화
    MAX_MEMORY_PER_GPU: str = "5GB"  # GTX 1660 Super용 메모리 제한
    
    def __post_init__(self):
        if self.FALLBACK_MODELS is None:
            self.FALLBACK_MODELS = [
                "microsoft/DialoGPT-medium",             # 6GB VRAM에 최적화
                "microsoft/DialoGPT-small",              # 더 가벼운 옵션
                "microsoft/BioGPT",                      # 의료 특화 모델 (더 큼)
                "gpt2"                                   # 최소 폴백 모델
            ]
        
        if self.MEDICAL_MODEL_OPTIONS is None:
            self.MEDICAL_MODEL_OPTIONS = [
                "microsoft/BioGPT",                      # 의료 전문 모델
                "microsoft/DialoGPT-medium",             # 대화형 모델
                "facebook/blenderbot-400M-distill",     # 경량 대화 모델
                "gpt2-medium"                            # 중간 크기 범용 모델
            ]


@dataclass 
class DatabaseConfig:
    """데이터베이스 설정"""
    CHROMA_DB_DIR: str = "my_chroma_db"
    COLLECTION_NAME: str = "medical_docs"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SEARCH_TOP_K: int = 2  # 5초 목표: 3 → 2로 검색 속도 향상


@dataclass
class ServerConfig:
    """서버 설정"""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = None
    
    def __post_init__(self):
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = ["*"]  # 개발 환경


@dataclass
class VerificationConfig:
    """검증 시스템 설정"""
    MIN_CONFIDENCE_THRESHOLD: float = 70.0
    MIN_KEYWORD_OVERLAP_RATIO: float = 0.2
    EMERGENCY_KEYWORDS: List[str] = None
    HIGH_RISK_KEYWORDS: List[str] = None
    MEDIUM_RISK_KEYWORDS: List[str] = None
    TERM_NORMALIZATION_JSON: str = ""
    ALLOW_ENGLISH_MEDICAL_TERMS: bool = True
    ALLOWED_MEDICAL_ABBREVIATIONS: List[str] = None
    
    def __post_init__(self):
        if self.EMERGENCY_KEYWORDS is None:
            self.EMERGENCY_KEYWORDS = ['응급', '위급', '심각', '생명', '위험']
        
        if self.HIGH_RISK_KEYWORDS is None:
            self.HIGH_RISK_KEYWORDS = [
                '자가치료', '병원 안가도', '약 안먹어도', '위험하지 않', 
                '괜찮다', '문제없다', '치료 안해도'
            ]
        
        if self.MEDIUM_RISK_KEYWORDS is None:
            self.MEDIUM_RISK_KEYWORDS = [
                '증상', '치료', '약물', '처방', '합병증'
            ]

        if self.ALLOWED_MEDICAL_ABBREVIATIONS is None:
            # 설정 기반 목록(필요 시 환경변수로 오버라이드). 대표적 의학 약어만 최소 제공
            self.ALLOWED_MEDICAL_ABBREVIATIONS = [
                'A형', 'B형', 'C형', 'HBV', 'HCV', 'HIV', 'HPV', 'DNA', 'RNA',
                'LDL', 'HDL', 'BMI', 'CT', 'MRI'
            ]


class Config:
    """통합 설정 클래스"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.database = DatabaseConfig()
        self.server = ServerConfig()
        self.verification = VerificationConfig()
        
        # 환경 변수로 설정 오버라이드
        self._load_from_env()
    
    def _load_from_env(self):
        """환경 변수에서 설정 로드"""
        # 서버 설정
        self.server.HOST = os.getenv("SERVER_HOST", self.server.HOST)
        self.server.PORT = int(os.getenv("SERVER_PORT", self.server.PORT))
        self.server.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        self.server.LOG_LEVEL = os.getenv("LOG_LEVEL", self.server.LOG_LEVEL)
        
        # 모델 설정
        self.model.PRIMARY_MODEL_ID = os.getenv("PRIMARY_MODEL_ID", self.model.PRIMARY_MODEL_ID)
        self.model.MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", self.model.MAX_NEW_TOKENS))
        self.model.TEMPERATURE = float(os.getenv("TEMPERATURE", self.model.TEMPERATURE))
        self.model.FORCE_RESPONSE_LANGUAGE = os.getenv("FORCE_RESPONSE_LANGUAGE", str(self.model.FORCE_RESPONSE_LANGUAGE)).lower() in ["1","true","yes"]
        self.model.RESPONSE_LANGUAGE = os.getenv("RESPONSE_LANGUAGE", self.model.RESPONSE_LANGUAGE)
        self.model.LANGUAGE_ENFORCEMENT_PROMPT = os.getenv("LANGUAGE_ENFORCEMENT_PROMPT", self.model.LANGUAGE_ENFORCEMENT_PROMPT)
        
        # 데이터베이스 설정
        self.database.CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", self.database.CHROMA_DB_DIR)

        # 검증/언어 관련 추가 설정
        self.verification.TERM_NORMALIZATION_JSON = os.getenv("TERM_NORMALIZATION_JSON", self.verification.TERM_NORMALIZATION_JSON)
        self.verification.ALLOW_ENGLISH_MEDICAL_TERMS = os.getenv("ALLOW_ENGLISH_MEDICAL_TERMS", str(self.verification.ALLOW_ENGLISH_MEDICAL_TERMS)).lower() in ["1","true","yes"]
        allowed_abbr_env = os.getenv("ALLOWED_MEDICAL_ABBREVIATIONS", "")
        if allowed_abbr_env:
            self.verification.ALLOWED_MEDICAL_ABBREVIATIONS = [x.strip() for x in allowed_abbr_env.split(',') if x.strip()]
    
    def get_model_options(self) -> List[str]:
        """모델 옵션 반환"""
        return [self.model.PRIMARY_MODEL_ID] + self.model.FALLBACK_MODELS
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "model": {
                "primary_model": self.model.PRIMARY_MODEL_ID,
                "embedding_model": self.model.EMBEDDING_MODEL_NAME,
                "max_tokens": self.model.MAX_NEW_TOKENS,
                "temperature": self.model.TEMPERATURE,
                "force_language": self.model.FORCE_RESPONSE_LANGUAGE,
                "response_language": self.model.RESPONSE_LANGUAGE
            },
            "database": {
                "db_dir": self.database.CHROMA_DB_DIR,
                "collection": self.database.COLLECTION_NAME
            },
            "server": {
                "host": self.server.HOST,
                "port": self.server.PORT,
                "debug": self.server.DEBUG
            },
            "verification": {
                "min_keyword_overlap_ratio": self.verification.MIN_KEYWORD_OVERLAP_RATIO,
                "allow_english_medical_terms": self.verification.ALLOW_ENGLISH_MEDICAL_TERMS,
                "allowed_medical_abbreviations": self.verification.ALLOWED_MEDICAL_ABBREVIATIONS
            }
        }


# 글로벌 설정 인스턴스
config = Config()
