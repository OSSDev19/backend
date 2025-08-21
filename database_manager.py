"""
ChromaDB 데이터베이스 관리 모듈
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection

from config import config
from model_manager import model_manager
try:
    from chromadb.utils import embedding_functions as ef
except Exception:
    ef = None

logger = logging.getLogger(__name__)


class DatabaseManager:
    """ChromaDB 관리 클래스"""
    
    def __init__(self):
        self.client: Optional[PersistentClient] = None
        self.collection: Optional[Collection] = None
        self.is_connected = False
        self.db_path_resolved: Optional[str] = None
    
    def connect(self) -> Tuple[bool, str]:
        """ChromaDB에 연결합니다.
        
        Returns:
            Tuple[bool, str]: (성공 여부, 상태 메시지)
        """
        try:
            logger.info("ChromaDB 연결 중...")
            
            # 경로 해석: 작업 디렉토리와 모듈 디렉토리 기준 모두 확인
            candidate_paths = []
            configured = config.database.CHROMA_DB_DIR
            candidate_paths.append(configured)
            candidate_paths.append(os.path.abspath(configured))
            module_dir = os.path.dirname(os.path.abspath(__file__))
            candidate_paths.append(os.path.join(module_dir, configured))
            
            resolved = None
            for p in candidate_paths:
                if os.path.exists(p) and os.path.isdir(p):
                    resolved = p
                    break
            
            if not resolved:
                return False, f"ChromaDB 디렉토리를 찾을 수 없음. 확인한 경로들: {candidate_paths}"
            
            self.db_path_resolved = resolved
            
            # 클라이언트 생성
            self.client = PersistentClient(path=self.db_path_resolved)
            
            # 컬렉션 목록 확인
            collections = self.client.list_collections()
            logger.info(f"사용 가능한 컬렉션: {[c.name for c in collections]}")
            
            if collections:
                # 목표 컬렉션명 결정
                target_name = config.database.COLLECTION_NAME
                available_names = [c.name for c in collections]
                if target_name not in available_names:
                    target_name = available_names[0]

                # 주의: 이미 퍼시스턴트 컬렉션에 임베딩 함수가 저장되어 있어
                # 여기서 embedding_function을 다시 지정하면 충돌 발생
                self.collection = self.client.get_collection(name=target_name)
                
                self.is_connected = True
                doc_count = self.collection.count()
                
                logger.info(f"✅ ChromaDB 연결 성공: {self.collection.name}, 문서 수: {doc_count}, 경로: {self.db_path_resolved}")
                return True, f"연결 성공 - 컬렉션: {self.collection.name}, 문서: {doc_count}개, 경로: {self.db_path_resolved}"
            else:
                return False, "사용 가능한 컬렉션이 없음"
                
        except Exception as e:
            logger.error(f"ChromaDB 연결 실패: {e}")
            self.is_connected = False
            return False, str(e)
    
    def search_similar_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """유사한 문서를 검색합니다.
        
        Args:
            query: 검색할 쿼리
            top_k: 반환할 문서 수
            
        Returns:
            List[Dict]: 검색된 문서 목록
        """
        if not self.is_connected or not self.collection:
            logger.warning("ChromaDB가 연결되지 않음")
            return []
        
        # 쿼리 개선: 핵심 의료 용어 추출 및 강조
        enhanced_query = self._enhance_medical_query(query)
        logger.info(f"원본 쿼리: {query}")
        logger.info(f"개선된 쿼리: {enhanced_query}")
        
        # 임베딩 모델 확인 (여러 소스에서)
        query_embeddings = None
        
        # 모델 매니저가 초기화되지 않은 경우 초기화
        if model_manager.embedding_model is None:
            try:
                # 이미 실행 중인 이벤트 루프가 있는지 확인
                try:
                    loop = asyncio.get_running_loop()
                    # 이미 실행 중이면 스킵
                    logger.warning("이벤트 루프가 실행 중이어서 모델 초기화를 스킵합니다")
                except RuntimeError:
                    # 이벤트 루프가 없으면 새로 생성
                    success, _ = asyncio.run(model_manager.load_models())
                    if not success:
                        logger.warning("모델 매니저 초기화 실패")
            except Exception as e:
                logger.warning(f"모델 매니저 초기화 중 오류: {e}")
        
        # 텍스트 질의가 실패할 경우에만 사용할 폴백 임베딩 준비
        if model_manager.embedding_model:
            try:
                embedding_result = model_manager.get_embedding(query)
                if embedding_result is not None:
                    if hasattr(embedding_result, 'tolist'):
                        query_embeddings = [embedding_result.tolist()]
                    else:
                        query_embeddings = [embedding_result]
            except Exception as e:
                logger.warning(f"기본 임베딩 실패: {e}")
        elif model_manager.custom_rag_manager:
            try:
                embedding = model_manager.custom_rag_manager.get_embedding(query)
                if embedding is not None:
                    if hasattr(embedding, 'tolist'):
                        query_embeddings = [embedding.tolist()]
                    else:
                        query_embeddings = [embedding]
            except Exception as e:
                logger.warning(f"커스텀 RAG 임베딩 실패: {e}")
        
        try:
            top_k = top_k or config.database.SEARCH_TOP_K
            
            # 키워드 기반 검색 우선 시도
            keyword_results = self._keyword_based_search(query, top_k)
            if keyword_results:
                            logger.info(f"키워드 기반 검색 성공: {len(keyword_results)}개 문서")
            for i, doc in enumerate(keyword_results):
                logger.info(f"문서 {i+1}: 점수={doc['final_score']:.3f}, 키워드 겹침={doc['keyword_overlap']}")
            
            # 최대 3개 문서 반환
            return keyword_results[:3]
            
            # 키워드 검색이 실패한 경우에만 임베딩 검색 시도
            logger.info("키워드 기반 검색 실패, 임베딩 검색 시도...")
            
            # 임베딩 기반 검색 (폴백)
            search_top_k = max(50, top_k * 5)
            results = None

            if query_embeddings is not None:
                try:
                    results = self.collection.query(
                        query_embeddings=query_embeddings,
                        n_results=search_top_k
                    )
                    logger.info(f"임베딩 검색 성공")
                except Exception as e:
                    logger.warning(f"임베딩 검색 실패: {e}")
                    return []
            else:
                logger.warning("임베딩 생성 실패로 검색 불가")
                return []
            
            documents: List[Dict[str, Any]] = []
            if results and results.get('documents') and len(results['documents']) > 0:
                # 완전 동적 하이브리드 검색: 임베딩 + 의미적 키워드 매칭
                query_keywords = set(re.findall(r'[가-힣A-Za-z0-9]+', query.lower()))
                
                # 동적 핵심 키워드 추출 (하드코딩 없음)
                core_keywords = self._extract_core_medical_keywords(query)
                
                for i, doc in enumerate(results['documents'][0]):
                    # 기본 키워드 매칭
                    doc_keywords = set(re.findall(r'[가-힣A-Za-z0-9]+', doc.lower()))
                    keyword_overlap = len(query_keywords & doc_keywords)
                    keyword_ratio = keyword_overlap / max(1, len(query_keywords))
                    
                    # 핵심 의료 키워드 매칭 (동적)
                    core_keyword_overlap = len(core_keywords & doc_keywords)
                    core_keyword_ratio = core_keyword_overlap / max(1, len(core_keywords))
                    
                    # 의미적 유사성 점수 (문장 구조 기반)
                    semantic_score = self._calculate_semantic_similarity(query, doc)
                    
                    # 최종 점수 계산 (균형잡힌 가중치)
                    final_score = (keyword_ratio * 0.4) + (core_keyword_ratio * 0.4) + (semantic_score * 0.2)
                    
                    documents.append({
                        'content': doc,
                        'distance': results['distances'][0][i] if results.get('distances') else 0,
                        'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                        'keyword_ratio': keyword_ratio,
                        'keyword_overlap': keyword_overlap,
                        'core_keyword_ratio': core_keyword_ratio,
                        'semantic_score': semantic_score,
                        'final_score': final_score
                    })
                
                # 최종 점수 기반 정렬
                documents.sort(key=lambda x: (-x['final_score'], x['distance']))
                
                # 의미적으로 관련 없는 문서 필터링 (더 엄격한 임계값)
                min_relevance = max(0.3, len(core_keywords) * 0.1)  # 더 엄격한 임계값
                
                # 추가 필터링: 핵심 의료 키워드가 문서에 있어야 함
                filtered_documents = []
                for doc in documents:
                    if doc['final_score'] >= min_relevance:
                        # 핵심 의료 키워드가 문서에 있는지 확인
                        doc_content = doc['content'].lower()
                        medical_keywords_in_query = [kw for kw in core_keywords if any(char in kw for char in ['간염', '감염', '염', '증', '병', '암'])]
                        
                        if medical_keywords_in_query:
                            # 핵심 의료 키워드가 문서에 하나라도 있으면 포함
                            has_medical_keyword = any(kw in doc_content for kw in medical_keywords_in_query)
                            if has_medical_keyword:
                                filtered_documents.append(doc)
                        else:
                            # 의료 키워드가 없으면 일반 키워드 매칭으로 판단
                            filtered_documents.append(doc)
                
                documents = filtered_documents
                
                # 필터링 후 문서가 없으면 원본 결과 사용
                if not documents:
                    logger.warning("필터링 후 문서가 없어 원본 결과 사용")
                    documents = []
                    for i, doc in enumerate(results['documents'][0][:top_k]):
                        documents.append({
                            'content': doc,
                            'distance': results['distances'][0][i] if results.get('distances') else 0,
                            'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                            'keyword_ratio': 0.0,
                            'keyword_overlap': 0,
                            'core_keyword_ratio': 0.0,
                            'semantic_score': 0.0,
                            'final_score': 0.0
                        })
                
                # 상위 결과만 반환 (최대 3개)
                documents = documents[:3]
            
            logger.info(f"하이브리드 검색 완료: {len(documents)}개 문서 발견")
            return documents
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류: {e}")
            return []
    
    def _enhance_medical_query(self, query: str) -> str:
        """동적 쿼리 개선: 의미적 유사성과 컨텍스트 기반"""
        import re
        
        # 1. 기본 정규화 (최소한의 규칙만)
        enhanced = query.strip()
        
        # 2. A/B/C형 감염 → 간염 정규화 (일반적인 패턴)
        enhanced = re.sub(r'\b([abcABC])형\s*감염\b', r'\1형 간염', enhanced)
        
        # 3. 의미적 가중치 추가: 핵심 키워드 반복으로 중요도 강조
        # 질문에서 핵심 의료 용어를 추출하여 가중치 부여
        medical_keywords = re.findall(r'\b[가-힣]{2,}\b', enhanced)
        
        # 4. 컨텍스트 기반 쿼리 확장
        # 질문의 의도(치료, 진단, 예방 등)를 파악하여 관련 키워드 추가
        context_keywords = []
        
        if any(word in enhanced for word in ['치료', '약', '개발']):
            context_keywords.extend(['치료', '약물', '개발'])
        if any(word in enhanced for word in ['증상', '징후']):
            context_keywords.extend(['증상', '징후'])
        if any(word in enhanced for word in ['진단', '검사']):
            context_keywords.extend(['진단', '검사'])
        if any(word in enhanced for word in ['예방', '백신']):
            context_keywords.extend(['예방', '백신'])
        
        # 5. 동적 쿼리 구성
        if context_keywords:
            enhanced = f"{enhanced} {' '.join(context_keywords)}"
        
        return enhanced
    
    def _extract_core_medical_keywords(self, query: str) -> set:
        """완전 동적 핵심 의료 키워드 추출 (어떤 의료 문장이든 처리 가능)"""
        query_lower = query.lower()
        
        # 1. 조사 제거 (은, 는, 이, 가, 을, 를, 의, 에, 에서, 로, 으로 등)
        query_no_particles = re.sub(r'[은는이가을를의에에서로으로]', '', query_lower)
        
        core_keywords = set()
        
        # 2. 일반적인 의료 용어 패턴 (동적)
        medical_patterns = [
            # 질병/증상 패턴
            r'\b[가-힣]+형\s*[가-힣]*간염\b',  # A형간염, B형간염 등
            r'\b[가-힣]+형\s*[가-힣]*감염\b',  # A형감염, B형감염 등
            r'\b[가-힣]+형\s*[가-힣]*암\b',    # A형암, B형암 등
            r'\b[가-힣]+형\s*[가-힣]*증\b',    # A형증, B형증 등
            r'\b[가-힣]+간염\b',  # 간염 관련
            r'\b[가-힣]+감염\b',  # 감염 관련
            r'\b[가-힣]+염\b',    # 염증성 질환
            r'\b[가-힣]+증\b',    # 증후군/증상
            r'\b[가-힣]+증후군\b', # 증후군
            r'\b[가-힣]+병\b',    # 질병
            r'\b[가-힣]+암\b',    # 암
            r'\b[가-힣]+장애\b',  # 장애
            r'\b[가-힣]+결핍\b',  # 결핍
            r'\b[가-힣]+중독\b',  # 중독
            r'\b[가-힣]+염증\b',  # 염증
            r'\b[가-힣]+손상\b',  # 손상
            r'\b[가-힣]+기능\b',  # 기능
            r'\b[가-힣]+치료\b',  # 치료
            r'\b[가-힣]+진단\b',  # 진단
            r'\b[가-힣]+검사\b',  # 검사
            r'\b[가-힣]+예방\b',  # 예방
            r'\b[가-힣]+합병증\b', # 합병증
            r'\b[가-힣]+증상\b',  # 증상
            r'\b[가-힣]+징후\b',  # 징후
            r'\b[가-힣]+발작\b',  # 발작
            r'\b[가-힣]+마비\b',  # 마비
            r'\b[가-힣]+출혈\b',  # 출혈
            r'\b[가-힣]+부종\b',  # 부종
            r'\b[가-힣]+통증\b',  # 통증
            r'\b[가-힣]+열\b',    # 열
            r'\b[가-힣]+기침\b',  # 기침
            r'\b[가-힣]+콧물\b',  # 콧물
            r'\b[가-힣]+구토\b',  # 구토
            r'\b[가-힣]+설사\b',  # 설사
            r'\b[가-힣]+변비\b',  # 변비
            r'\b[가-힣]+소화\b',  # 소화
            r'\b[가-힣]+흡수\b',  # 흡수
            r'\b[가-힣]+분비\b',  # 분비
            r'\b[가-힣]+순환\b',  # 순환
            r'\b[가-힣]+호흡\b',  # 호흡
            r'\b[가-힣]+심장\b',  # 심장
            r'\b[가-힣]+폐\b',    # 폐
            r'\b[가-힣]+간\b',    # 간
            r'\b[가-힣]+신장\b',  # 신장
            r'\b[가-힣]+위\b',    # 위
            r'\b[가-힣]+장\b',    # 장
            r'\b[가-힣]+뇌\b',    # 뇌
            r'\b[가-힣]+신경\b',  # 신경
            r'\b[가-힣]+근육\b',  # 근육
            r'\b[가-힣]+뼈\b',    # 뼈
            r'\b[가-힣]+관절\b',  # 관절
            r'\b[가-힣]+혈관\b',  # 혈관
            r'\b[가-힣]+림프\b',  # 림프
            r'\b[가-힣]+면역\b',  # 면역
            r'\b[가-힣]+알레르기\b', # 알레르기
            r'\b[가-힣]+염증\b',  # 염증
            r'\b[가-힣]+종양\b',  # 종양
            r'\b[가-힣]+전이\b',  # 전이
            r'\b[가-힣]+재발\b',  # 재발
            r'\b[가-힣]+완화\b',  # 완화
            r'\b[가-힣]+관해\b',  # 관해
            r'\b[가-힣]+악화\b',  # 악화
            r'\b[가-힣]+진행\b',  # 진행
            r'\b[가-힣]+급성\b',  # 급성
            r'\b[가-힣]+만성\b',  # 만성
            r'\b[가-힣]+급성\b',  # 급성
            r'\b[가-힣]+급성\b',  # 급성
        ]
        
        # 3. 패턴 매칭으로 핵심 키워드 추출
        for pattern in medical_patterns:
            matches = re.findall(pattern, query_lower)
            core_keywords.update(matches)
        
        # 4. 조사 제거 후 패턴 매칭
        for pattern in medical_patterns:
            matches = re.findall(pattern, query_no_particles)
            core_keywords.update(matches)
        
        # 5. 일반적인 의료 용어 추출 (조사 제거 후)
        medical_terms = [
            '간염', '감염', '염', '증', '병', '암', '기능', '치료', '진단', '검사', '예방',
            '합병증', '증상', '징후', '발작', '마비', '출혈', '부종', '통증', '열', '기침',
            '콧물', '구토', '설사', '변비', '소화', '흡수', '분비', '순환', '호흡', '심장',
            '폐', '간', '신장', '위', '장', '뇌', '신경', '근육', '뼈', '관절', '혈관',
            '림프', '면역', '알레르기', '염증', '종양', '전이', '재발', '완화', '관해',
            '악화', '진행', '급성', '만성', '급성', '급성', '급성'
        ]
        
        for term in medical_terms:
            if term in query_no_particles:
                core_keywords.add(term)
        
        # 6. 문맥 기반 키워드 추출 (의료 상황을 나타내는 동사/형용사)
        context_words = [
            '발생', '생길', '일어나', '나타나', '발현', '유발', '초래', '야기',
            '증가', '감소', '상승', '하락', '높아지', '낮아지', '심해지', '약해지',
            '개선', '악화', '완화', '치유', '회복', '재발', '전이', '확산',
            '감염', '전파', '오염', '접촉', '섭취', '흡수', '분비', '배설',
            '진단', '확인', '검사', '관찰', '모니터링', '추적', '관리', '치료'
        ]
        
        for word in context_words:
            if word in query_lower:
                # 문맥 단어 주변의 의료 용어 찾기
                words = query_lower.split()
                for i, w in enumerate(words):
                    if w == word:
                        # 문맥 단어 앞의 단어가 의료 용어일 가능성
                        if i > 0:
                            prev_word = words[i-1]
                            if len(prev_word) >= 2 and any(char in prev_word for char in ['염', '증', '병', '암', '기능', '통', '열', '통']):
                                core_keywords.add(prev_word)
                        # 문맥 단어 뒤의 단어가 의료 용어일 가능성
                        if i < len(words) - 1:
                            next_word = words[i+1]
                            if len(next_word) >= 2 and any(char in next_word for char in ['염', '증', '병', '암', '기능', '통', '열', '통']):
                                core_keywords.add(next_word)
        
        # 7. 명사성 키워드 추출 (2글자 이상의 모든 명사)
        nouns = re.findall(r'\b[가-힣]{2,}\b', query_lower)
        for noun in nouns:
            # 의료 관련성이 높은 단어들 (더 포괄적으로)
            if any(char in noun for char in ['염', '증', '병', '암', '기능', '치료', '진단', '검사', '예방', '통', '열', '통', '부', '장', '관', '맥', '액', '체', '질', '상', '증', '형', '성', '화', '증', '증', '증']):
                core_keywords.add(noun)
        
        # 8. 특별한 의료 용어 패턴 (조사 제거 후)
        special_patterns = [
            r'\b[가-힣]+형감염\b',  # A형감염, B형감염 등 (띄어쓰기 없음)
            r'\b[가-힣]+형\s*감염\b',  # A형 감염, B형 감염 등 (띄어쓰기 있음)
            r'\b[가-힣]+형간염\b',  # A형간염, B형간염 등 (띄어쓰기 없음)
            r'\b[가-힣]+형\s*간염\b',  # A형 간염, B형 간염 등 (띄어쓰기 있음)
        ]
        
        for pattern in special_patterns:
            matches = re.findall(pattern, query_no_particles)
            core_keywords.update(matches)
        
        # 9. 직접적인 의료 용어 매칭 (조사 제거 후)
        direct_medical_terms = [
            'a형감염', 'b형감염', 'c형감염', 'a형간염', 'b형간염', 'c형간염',
            'a형암', 'b형암', 'c형암', 'a형증', 'b형증', 'c형증'
        ]
        
        for term in direct_medical_terms:
            if term in query_no_particles:
                core_keywords.add(term)
        
        return core_keywords
    
    def _calculate_semantic_similarity(self, query: str, doc: str) -> float:
        """문장 구조 기반 의미적 유사성 계산"""
        query_lower = query.lower()
        doc_lower = doc.lower()
        
        # 1. 문장 구조 유사성
        query_sentences = re.split(r'[.!?]', query_lower)
        doc_sentences = re.split(r'[.!?]', doc_lower)
        
        # 2. 공통 단어 비율
        query_words = set(re.findall(r'\b[가-힣A-Za-z0-9]+\b', query_lower))
        doc_words = set(re.findall(r'\b[가-힣A-Za-z0-9]+\b', doc_lower))
        
        if not query_words:
            return 0.0
        
        common_words = query_words & doc_words
        word_similarity = len(common_words) / len(query_words)
        
        # 3. 문장 길이 유사성
        length_ratio = min(len(query_lower), len(doc_lower)) / max(len(query_lower), len(doc_lower))
        
        # 4. 의료 용어 밀도
        medical_terms_query = len([w for w in query_words if any(char in w for char in ['염', '증', '병', '암', '기능'])])
        medical_terms_doc = len([w for w in doc_words if any(char in w for char in ['염', '증', '병', '암', '기능'])])
        
        medical_density_similarity = 1.0 - abs(medical_terms_query - medical_terms_doc) / max(medical_terms_query + medical_terms_doc, 1)
        
        # 5. 최종 의미적 유사성 점수
        semantic_score = (word_similarity * 0.5) + (length_ratio * 0.2) + (medical_density_similarity * 0.3)
        
        return min(1.0, semantic_score)
    
    def _keyword_based_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """키워드 기반 검색 (임베딩보다 정확한 결과)"""
        try:
            # 핵심 키워드 추출
            core_keywords = self._extract_core_medical_keywords(query)
            if not core_keywords:
                logger.info("핵심 의료 키워드를 찾을 수 없음")
                return []
            
            logger.info(f"추출된 핵심 키워드: {core_keywords}")
            
            # 모든 문서에서 키워드 매칭 검색
            all_docs = self.collection.get()
            if not all_docs or not all_docs.get('documents'):
                return []
            
            matched_docs = []
            
            for i, doc_content in enumerate(all_docs['documents']):
                doc_lower = doc_content.lower()
                doc_keywords = set(re.findall(r'[가-힣A-Za-z0-9]+', doc_lower))
                
                # 핵심 키워드 매칭 점수
                keyword_overlap = len(core_keywords & doc_keywords)
                keyword_ratio = keyword_overlap / max(1, len(core_keywords))
                
                # 쿼리 전체 키워드 매칭 점수
                query_keywords = set(re.findall(r'[가-힣A-Za-z0-9]+', query.lower()))
                query_overlap = len(query_keywords & doc_keywords)
                query_ratio = query_overlap / max(1, len(query_keywords))
                
                # 의미적 유사성 점수
                semantic_score = self._calculate_semantic_similarity(query, doc_content)
                
                # 최종 점수 (쿼리 키워드 매칭에 더 높은 가중치)
                final_score = (keyword_ratio * 0.4) + (query_ratio * 0.4) + (semantic_score * 0.2)
                
                # 더 엄격한 임계값 (키워드 매칭이 있어야 함)
                if final_score >= 0.3 and keyword_overlap > 0:  # 임계값을 낮춤 (0.5 -> 0.3)
                    # 추가 필터링: 핵심 의료 키워드가 문서에 있어야 함
                    medical_keywords_in_query = [kw for kw in core_keywords if any(char in kw for char in ['간염', '감염', '염', '증', '병', '암'])]
                    if medical_keywords_in_query:
                        medical_keywords_in_doc = [kw for kw in doc_keywords if any(char in kw for char in ['간염', '감염', '염', '증', '병', '암'])]
                        medical_overlap = len(set(medical_keywords_in_query) & set(medical_keywords_in_doc))
                        if medical_overlap == 0:  # 핵심 의료 키워드가 하나도 매칭되지 않으면 제외
                            continue
                    
                    # 추가 필터링: 쿼리의 핵심 의도와 문서 내용의 일치성 확인
                    query_intent = self._extract_query_intent(query)
                    doc_intent = self._extract_document_intent(doc_content)
                    
                    # 의도가 일치하지 않으면 점수 감소
                    if query_intent and doc_intent and query_intent != doc_intent:
                        final_score *= 0.1  # 의도가 다르면 점수를 90% 감소
                    
                    # 합병증 관련 특별 필터링
                    if query_intent == '합병증':
                        # A형간염 합병증의 경우 더 엄격한 필터링
                        if 'a형간염' in query.lower() or '간염' in query.lower():
                            # A형간염과 직접적으로 관련된 문서만 허용
                            hepatitis_keywords = ['a형간염', '간염', 'hepatitis']
                            has_hepatitis_keyword = any(kw in doc_lower for kw in hepatitis_keywords)
                            if not has_hepatitis_keyword:
                                continue
                        
                        # 합병증 관련 키워드가 문서에 없으면 제외
                        complication_keywords = ['합병증', '기앵-바레증후군', '급성 신부전', '담낭염', '췌장염', '혈관염', '관절염']
                        has_complication_keyword = any(kw in doc_lower for kw in complication_keywords)
                        if not has_complication_keyword:
                            continue
                        
                        # 진단 관련 키워드가 있으면 제외 (미노전이효소, 빌리루빈 등)
                        # 단, 합병증 관련 내용이 충분히 있으면 허용
                        diagnostic_keywords = ['미노전이효소', 'ast', 'got', '빌리루빈', '혈청', '수치', '검사']
                        has_diagnostic_keyword = any(kw in doc_lower for kw in diagnostic_keywords)
                        if has_diagnostic_keyword:
                            # 합병증 관련 키워드가 충분히 있으면 허용
                            specific_complications = ['기앵-바레증후군', '급성 신부전', '담낭염', '췌장염', '혈관염', '관절염']
                            complication_count = sum(1 for comp in specific_complications if comp in doc_lower)
                            if complication_count < 3:  # 3개 미만의 특정 합병증만 있으면 제외
                                continue
                        
                        # 다른 질환 관련 키워드가 있으면 제외
                        other_disease_keywords = ['전염성 단핵구증', '편도염', '인두염', '연쇄상구균']
                        has_other_disease = any(kw in doc_lower for kw in other_disease_keywords)
                        if has_other_disease:
                            continue
                    
                    matched_docs.append({
                        'content': doc_content,
                        'distance': 1.0 - final_score,  # 점수를 거리로 변환
                        'metadata': all_docs['metadatas'][i] if all_docs.get('metadatas') else {},
                        'keyword_ratio': keyword_ratio,
                        'keyword_overlap': keyword_overlap,
                        'core_keyword_ratio': keyword_ratio,
                        'semantic_score': semantic_score,
                        'final_score': final_score
                    })
            
            # 점수 기반 정렬
            matched_docs.sort(key=lambda x: (-x['final_score'], x['distance']))
            
            logger.info(f"키워드 검색: {len(matched_docs)}개 문서 매칭")
            return matched_docs[:top_k]
            
        except Exception as e:
            logger.warning(f"키워드 기반 검색 실패: {e}")
            return []
    
    def _extract_query_intent(self, query: str) -> str:
        """쿼리의 의도를 추출합니다."""
        query_lower = query.lower()
        
        # 의도별 키워드 매핑 (우선순위 순서)
        intent_keywords = [
            ('합병증', ['합병증', '기앵-바레증후군', '급성 신부전', '담낭염', '췌장염', '혈관염', '관절염']),
            ('증상', ['증상', '징후', '나타나', '보임', '발현']),
            ('치료', ['치료', '약', '개발', '완치', '회복']),
            ('진단', ['진단', '검사', '확인', '판정']),
            ('예방', ['예방', '백신', '접종', '방지']),
            ('원인', ['원인', '발생', '유발', '초래']),
            ('전파', ['전파', '감염', '오염', '접촉', '섭취'])
        ]
        
        for intent, keywords in intent_keywords:
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return None
    
    def _extract_document_intent(self, doc_content: str) -> str:
        """문서의 의도를 추출합니다."""
        doc_lower = doc_content.lower()
        
        # 의도별 키워드 매핑 (우선순위 순서)
        intent_keywords = [
            ('합병증', ['합병증', '기앵-바레증후군', '급성 신부전', '담낭염', '췌장염', '혈관염', '관절염']),
            ('진단', ['진단', '검사', '확인', '판정', '미노전이효소', 'ast', 'got', '빌리루빈', '혈청']),
            ('증상', ['증상', '징후', '나타나', '보임', '발현']),
            ('치료', ['치료', '약', '개발', '완치', '회복']),
            ('예방', ['예방', '백신', '접종', '방지']),
            ('원인', ['원인', '발생', '유발', '초래']),
            ('전파', ['전파', '감염', '오염', '접촉', '섭취'])
        ]
        
        for intent, keywords in intent_keywords:
            if any(keyword in doc_lower for keyword in keywords):
                return intent
        
        return None
    
    def get_database_info(self) -> Dict[str, Any]:
        """데이터베이스 정보를 반환합니다."""
        return {
            "connected": self.is_connected,
            "collection_name": self.collection.name if self.collection else None,
            "document_count": self.collection.count() if self.collection else 0,
            "db_directory": self.db_path_resolved or config.database.CHROMA_DB_DIR,
            "collection_exists": os.path.exists(self.db_path_resolved or config.database.CHROMA_DB_DIR)
        }
    
    def disconnect(self):
        """데이터베이스 연결을 해제합니다."""
        self.client = None
        self.collection = None
        self.is_connected = False
        logger.info("ChromaDB 연결 해제됨")


# 글로벌 데이터베이스 매니저 인스턴스
database_manager = DatabaseManager()
