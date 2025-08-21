"""
AI 모델 관리 모듈 - Ollama + 커스텀 모델 통합 버전
"""

import logging
import aiohttp
import asyncio
from typing import Optional, Tuple, Dict, Any
import warnings

from config import config

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class ModelManager:
    """Ollama + 커스텀 모델 통합 관리 클래스"""
    
    def __init__(self):
        self.ollama_available = False
        self.custom_rag_manager = None
        self.embedding_model = None
        self.current_model_type = None  # "ollama" or "custom"
        
    async def load_models(self) -> Tuple[bool, str]:
        """
        설정에 따라 Ollama 또는 커스텀 모델을 로드합니다.
        
        Returns:
            Tuple[bool, str]: (성공 여부, 상태 메시지)
        """
        try:
            logger.info("모델 로딩 시작...")
            
            # 1. 커스텀 모델 사용 설정인 경우
            if config.model.USE_CUSTOM_MODEL and config.model.CUSTOM_MODEL_PATH:
                success, msg = self._load_custom_model()
                if success:
                    self.current_model_type = "custom"
                    logger.info("✅ 커스텀 모델 로딩 성공!")
                else:
                    logger.warning(f"⚠️ 커스텀 모델 로딩 실패: {msg}")
                    # 폴백: Ollama 시도
                    if config.model.USE_OLLAMA:
                        return await self._try_ollama_fallback()
                    else:
                        return False, f"커스텀 모델 로딩 실패: {msg}"
            
            # 2. Ollama 사용 설정인 경우
            elif config.model.USE_OLLAMA:
                ollama_success, ollama_msg = await self._check_ollama_connection()
                if ollama_success:
                    self.ollama_available = True
                    self.current_model_type = "ollama"
                    logger.info("✅ Ollama 연결 성공!")
                else:
                    logger.warning(f"⚠️ Ollama 연결 실패: {ollama_msg}")
                    # 폴백: 커스텀 모델 시도
                    if config.model.CUSTOM_MODEL_PATH:
                        return await self._try_custom_fallback()
                    else:
                        return False, f"Ollama 연결 실패: {ollama_msg}"
            
            else:
                return False, "Ollama와 커스텀 모델이 모두 비활성화됨"
            
            # 3. 임베딩 모델 로드 (공통)
            embedding_success, embedding_msg = self._load_embedding_model()
            if embedding_success:
                logger.info("✅ 임베딩 모델 로딩 성공!")
            else:
                logger.warning(f"⚠️ 임베딩 모델 로딩 실패: {embedding_msg}")
            
            success_msg = f"모델 매니저 초기화 완료 (타입: {self.current_model_type})"
            return True, success_msg
            
        except Exception as e:
            logger.error(f"모델 로딩 중 오류: {e}")
            return False, str(e)
    
    def _load_custom_model(self) -> Tuple[bool, str]:
        """커스텀 모델 로드"""
        try:
            # 동적 import (선택적 의존성)
            try:
                from custom_rag_manager import CustomRAGManager
            except ImportError:
                return False, "custom_rag_manager 모듈을 찾을 수 없음"
            
            self.custom_rag_manager = CustomRAGManager(config.model.CUSTOM_MODEL_PATH)
            
            # 모델 로드
            success, msg = self.custom_rag_manager.load_custom_model(config.model.CUSTOM_MODEL_PATH)
            if success:
                # 임베딩 모델도 로드
                self.custom_rag_manager.load_embedding_model()
                return True, msg
            else:
                return False, msg
            
        except Exception as e:
            logger.error(f"커스텀 모델 로딩 중 오류: {e}")
            return False, str(e)
    
    async def _try_ollama_fallback(self) -> Tuple[bool, str]:
        """Ollama 폴백 시도"""
        logger.info("Ollama 폴백 시도...")
        ollama_success, ollama_msg = await self._check_ollama_connection()
        if ollama_success:
            self.ollama_available = True
            self.current_model_type = "ollama"
            return True, "폴백으로 Ollama 연결 성공"
        else:
            return False, f"폴백 실패: {ollama_msg}"
    
    async def _try_custom_fallback(self) -> Tuple[bool, str]:
        """커스텀 모델 폴백 시도"""
        logger.info("커스텀 모델 폴백 시도...")
        success, msg = self._load_custom_model()
        if success:
            self.current_model_type = "custom"
            return True, "폴백으로 커스텀 모델 로딩 성공"
        else:
            return False, f"폴백 실패: {msg}"
    
    async def _check_ollama_connection(self) -> Tuple[bool, str]:
        """Ollama 서버 연결 상태를 확인합니다."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{config.model.OLLAMA_BASE_URL}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        
                        if config.model.OLLAMA_MODEL in models:
                            return True, f"모델 '{config.model.OLLAMA_MODEL}' 사용 가능"
                        else:
                            return False, f"모델 '{config.model.OLLAMA_MODEL}'이 설치되지 않음. 설치된 모델: {models}"
                    else:
                        return False, f"Ollama 서버 응답 오류: {response.status}"
                        
        except asyncio.TimeoutError:
            return False, "Ollama 서버 연결 시간 초과 (5초)"
        except aiohttp.ClientError as e:
            return False, f"Ollama 서버 연결 실패: {e}"
        except Exception as e:
            return False, f"Ollama 연결 확인 중 오류: {e}"
    
    def _load_embedding_model(self) -> Tuple[bool, str]:
        """임베딩 모델을 로드합니다. (선택사항)"""
        try:
            # 임베딩 모델은 sentence-transformers가 필요하므로 선택사항으로 처리
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.warning("sentence-transformers가 설치되지 않음. 임베딩 기능 비활성화")
                return False, "sentence-transformers 미설치"

            # 후보 모델들을 순차 시도 (768차원 우선)
            candidate_models = [
                config.model.EMBEDDING_MODEL_NAME,
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # 768차원
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384차원
                "sentence-transformers/all-MiniLM-L6-v2"  # 384차원
            ]

            last_err = None
            for model_name in candidate_models:
                try:
                    logger.info(f"임베딩 모델 로딩 시도: {model_name}")
                    self.embedding_model = SentenceTransformer(model_name)
                    logger.info(f"임베딩 모델 로딩 성공: {model_name}")
                    return True, f"임베딩 모델 로딩 성공: {model_name}"
                except Exception as e:
                    last_err = e
                    logger.warning(f"임베딩 모델 로딩 실패({model_name}): {e}")
                    continue

            return False, f"모든 임베딩 후보 로딩 실패: {last_err}"
                
        except Exception as e:
            logger.error(f"임베딩 모델 로딩 중 오류: {e}")
            return False, str(e)
    
    async def generate_response(self, prompt: str, system_prompt: str = "", context: str = "", 
                               max_new_tokens: int = None, temperature: float = None, 
                               do_sample: bool = None, repetition_penalty: float = None) -> Dict[str, Any]:
        """
        현재 활성화된 모델을 사용하여 응답을 생성합니다. (Colab 원본 매개변수 지원)
        
        Args:
            prompt: 사용자 질문
            system_prompt: 시스템 프롬프트
            context: 검색된 문서 컨텍스트 (RAG용)
            max_new_tokens: 최대 생성 토큰 수 (Colab 원본 지원)
            temperature: 생성 온도 (Colab 원본 지원)  
            do_sample: 샘플링 여부 (Colab 원본 지원)
            repetition_penalty: 반복 페널티 (Colab 원본 지원)
            
        Returns:
            Dict containing response and metadata
        """
        # 언어 강제 시스템 프롬프트 (설정 기반)
        enforced_system_prompt = system_prompt
        if config.model.FORCE_RESPONSE_LANGUAGE:
            lang = (config.model.RESPONSE_LANGUAGE or "").lower()
            if not config.model.LANGUAGE_ENFORCEMENT_PROMPT:
                if lang == "ko":
                    # 설정 기반 허용 정책 적용
                    allowed = ", ".join(getattr(config.verification, 'ALLOWED_MEDICAL_ABBREVIATIONS', []) or [])
                    allow_eng_terms = getattr(config.verification, 'ALLOW_ENGLISH_MEDICAL_TERMS', True)
                    eng_clause = ("의학적 약어/용어(예: " + allowed + ")는 그대로 사용해도 됩니다.") if allow_eng_terms else "영문 약어는 사용하지 않습니다."
                    enforce = (
                        "다음 지시를 반드시 따르세요:\n"
                        "1) 모든 출력은 한국어로 작성합니다.\n"
                        "2) 한국어 표기를 우선 사용합니다.\n"
                        f"3) {eng_clause}\n"
                        "4) 번역이 필요한 경우에도 결과는 한국어로 제시합니다."
                    )
                else:
                    enforce = f"Always respond strictly in the language: {lang}."
            else:
                enforce = config.model.LANGUAGE_ENFORCEMENT_PROMPT

            enforced_system_prompt = f"{enforce}\n\n{system_prompt}" if system_prompt else enforce

        if self.current_model_type == "custom" and self.custom_rag_manager:
            # 커스텀 모델 사용
            return await self.custom_rag_manager.generate_medical_response(
                query=prompt,
                context=context,
                system_prompt=enforced_system_prompt,
                max_new_tokens=config.model.MAX_NEW_TOKENS,
                temperature=config.model.TEMPERATURE
            )
        
        elif self.current_model_type == "ollama" and self.ollama_available:
            # Ollama 사용 (Colab 원본 매개변수 전달)
            return await self._generate_ollama_response(
                prompt, enforced_system_prompt, max_new_tokens, temperature, do_sample, repetition_penalty
            )
            
        else:
            return {
                "response": "사용 가능한 모델이 없습니다.",
                "error": "No model available",
                "success": False
            }
    
    async def _generate_ollama_response(self, prompt: str, system_prompt: str = "", 
                                       max_new_tokens: int = None, temperature: float = None,
                                       do_sample: bool = None, repetition_penalty: float = None) -> Dict[str, Any]:
        """Ollama API를 사용하여 응답 생성 (Colab 원본 매개변수 지원)"""
        
        try:
            # Colab 원본 매개변수 또는 기본값 사용
            final_temperature = temperature if temperature is not None else config.model.TEMPERATURE
            final_max_tokens = max_new_tokens if max_new_tokens is not None else config.model.MAX_NEW_TOKENS
            
            # Ollama API 호출을 위한 페이로드 (Colab 원본 매개변수 반영)
            options = {
                "temperature": final_temperature,
                "num_predict": final_max_tokens
            }
            
            # do_sample=False일 때는 온도를 0으로 설정 (Colab 원본 방식)
            if do_sample is False:
                options["temperature"] = 0.0
            
            # repetition_penalty 지원 (Ollama에서 지원하는 경우)
            if repetition_penalty is not None:
                options["repeat_penalty"] = repetition_penalty
            
            payload = {
                "model": config.model.OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                "stream": False,
                "options": options
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{config.model.OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                    timeout=300  # 5분으로 확장 (한국어 모델용)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "response": data.get("response", ""),
                            "success": True,
                            "model": config.model.OLLAMA_MODEL,
                            "tokens_used": data.get("eval_count", 0)
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "response": "Ollama API 호출 실패",
                            "error": f"HTTP {response.status}: {error_text}",
                            "success": False
                        }
                        
        except asyncio.TimeoutError:
            return {
                "response": "응답 생성 시간이 초과되었습니다.",
                "error": "시간 초과",
                "success": False
            }
        except Exception as e:
            logger.error(f"Ollama 응답 생성 중 오류: {e}")
            return {
                "response": "응답 생성 중 오류가 발생했습니다.",
                "error": str(e),
                "success": False
            }
    
    def get_embedding(self, text: str):
        """텍스트 임베딩을 생성합니다."""
        if self.embedding_model is None:
            logger.warning("임베딩 모델이 로드되지 않음")
            return None
        
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            logger.error(f"임베딩 생성 오류: {e}")
            return None
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보를 반환합니다."""
        info = {
            "current_model_type": self.current_model_type,
            "ollama_available": self.ollama_available,
            "custom_model_available": self.custom_rag_manager is not None,
            "embedding_available": self.embedding_model is not None,
            "config": {
                "use_ollama": config.model.USE_OLLAMA,
                "use_custom_model": config.model.USE_CUSTOM_MODEL,
                "custom_model_path": config.model.CUSTOM_MODEL_PATH
            }
        }
        
        # Ollama 정보 추가
        if self.ollama_available:
            info["ollama_info"] = {
                "url": config.model.OLLAMA_BASE_URL,
                "model": config.model.OLLAMA_MODEL
            }
        
        # 커스텀 모델 정보 추가
        if self.custom_rag_manager:
            info["custom_model_info"] = self.custom_rag_manager.get_system_info()
        
        return info


# 전역 모델 매니저 인스턴스
model_manager = ModelManager()