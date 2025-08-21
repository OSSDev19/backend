"""
API 핸들러 모듈 (RAG 기반)
"""

import logging
import re
from typing import Dict, Any, Set
from fastapi import HTTPException

from config import config
from model_manager import model_manager
from database_manager import database_manager
from medical_verification import MedicalFactChecker, format_verification_report

logger = logging.getLogger(__name__)

# 의료 정보 검증기 인스턴스
fact_checker = MedicalFactChecker()


class APIHandlers:
    """API 핸들러 클래스 (RAG 기반)"""
    
    @staticmethod
    async def health_check() -> Dict[str, Any]:
        """시스템 상태 및 정보 확인"""
        try:
            system_info = {
                "status": "healthy",
                "system_info": {
                    "models": model_manager.get_system_info(),
                    "database": database_manager.get_database_info(),
                    "verification_system": {
                        "fact_checker_active": fact_checker is not None,
                        "verification_features": [
                            "의료 정보 정확도 평가",
                            "위험도 분석", 
                            "근거 기반 검증",
                            "전문가 조언 제공"
                        ]
                    },
                    "configuration": config.to_dict()
                }
            }
            
            return system_info
            
        except Exception as e:
            logger.error(f"헬스 체크 중 오류: {e}")
            raise HTTPException(status_code=500, detail=f"시스템 상태 확인 실패: {str(e)}")
    
    @staticmethod
    async def process_query(query: str) -> Dict[str, Any]:
        """쿼리를 처리하고 의료 정보 검증 결과를 반환합니다. (RAG 기반)"""
        try:
            query = query.strip()
            if not query:
                raise HTTPException(status_code=400, detail="쿼리가 비어있습니다.")
            
            logger.info(f"쿼리 처리 중: {query[:100]}...")
            
            # 1. 유사 문서 검색
            similar_docs = database_manager.search_similar_documents(query)

            # 2. 의료 정보 검증 수행
            logger.info("의료 정보 검증 시작...")
            verification_result = await fact_checker.analyze_medical_claim(query, similar_docs)
            
            # 3. 전문적인 검증 보고서 생성
            answer = format_verification_report(verification_result, query)
            
            # 4. RAG 기반 추가 의학 정보 생성: '맞음/틀림'에서만 생성 (애매/민간요법 제외)
            if similar_docs and verification_result.judgment in ("올바른 의료 정보", "올바르지 않은 정보"):
                additional_info = await APIHandlers._generate_additional_medical_info(query, similar_docs)
                fallback_needed = (not additional_info) or (additional_info.strip() == "관련된 의학 정보를 찾을 수 없습니다.")
                if fallback_needed:
                    # 4-1. 보강: corrected_info 기반 최소 1~2문장 생성
                    fallback_text = (verification_result.corrected_info or "").strip()
                    # 너무 일반적인 문구는 제외하고, 문장 단위 정리
                    def _sanitize_and_clip(txt: str) -> str:
                        import re as _re
                        s = (txt or "").strip()
                        # 일본어/태국어 등 비한글 스크립트 제거 및 과도 공백 정리
                        s = _re.sub(r"[\u3040-\u30FF\u31F0-\u31FF\uFF65-\uFF9F\u0E00-\u0E7F]+", '', s)
                        s = _re.sub(r"\s{2,}", ' ', s)
                        # 문장 단위 분리 후 1~2문장만 유지
                        parts = [p.strip() for p in _re.split(r"[\.!?]\s+", s) if p.strip()]
                        picked = []
                        for p in parts:
                            if 12 <= len(p) <= 220:
                                picked.append(p)
                            if len(picked) >= 2:
                                break
                        if not picked and s:
                            picked = parts[:1] if parts else [s[:180]]
                        out = '. '.join([p.rstrip(' .') for p in picked])
                        if out and not _re.search(r"[\.!?]$", out):
                            out += '.'
                        # 어절 이상 공백 교정
                        out = _re.sub(r'있\s*습\s*니\s*다', '있습니다', out)
                        out = _re.sub(r'없\s*습\s*니\s*다', '없습니다', out)
                        out = _re.sub(r'됩\s*니\s*다', '됩니다', out)
                        return out.strip()

                    # 진단/무관 도메인 키워드 제외 세트
                    _diag = ['미노전이효소', 'ast', 'got', '빌리루빈', '혈청', '수치', '황달', '검사']
                    _other = ['곤충', '흡혈', '유충', '성충', '벼룩', '진드기', '모기']

                    if fallback_text and '관련' not in fallback_text:
                        tmp = _sanitize_and_clip(fallback_text)
                        low = tmp.lower()
                        if not any(k in low for k in _diag) and not any(k in low for k in _other):
                            additional_info = tmp
                    else:
                        # 4-2. 근거 문서에서 직접 1~2문장 추출 (질의 토큰과의 관련성 기준)
                        try:
                            import re as _re
                            q_tokens = set(_re.findall(r"[가-힣A-Za-z0-9]+", (query or '').lower()))
                            content = (similar_docs[0].get('content') or '')
                            sentences = [s.strip() for s in _re.split(r"[\.!?]\s+", content) if s.strip()]
                            picked = []
                            for s in sentences:
                                s_lower = s.lower()
                                s_tokens = set(_re.findall(r"[가-힣A-Za-z0-9]+", s_lower))
                                overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
                                # A형간염 맥락 강제 및 B/C형 간염 배제
                                has_hep_a = any(tok in s_lower for tok in ['a형간염', 'a형 간염', 'a형감염', 'hepatitis a', 'hav'])
                                has_hep_bc = any(tok in s_lower for tok in ['b형간염', 'b 형 간염', 'b형 간염', 'hbv', 'hepatitis b',
                                                                              'c형간염', 'c 형 간염', 'c형 간염', 'hcv', 'hepatitis c'])
                                has_diag = any(k in s_lower for k in _diag)
                                has_other = any(k in s_lower for k in _other)
                                if overlap >= 0.25 and 12 <= len(s) <= 220 and has_hep_a and not has_hep_bc and not has_diag and not has_other:
                                    picked.append(s)
                                if len(picked) >= 2:
                                    break
                            if picked:
                                tmp = '. '.join([p.rstrip(' .') for p in picked])
                                if tmp and not _re.search(r"[\.!?]$", tmp):
                                    tmp += '.'
                                additional_info = _sanitize_and_clip(tmp)
                        except Exception:
                            pass

                if additional_info:
                    answer += f"\n\n📚 추가 의학 정보:\n{additional_info}"

            # 4-β. 핵심 설명 품질 보강: 너무 일반적이면 RAG로 대체 생성
            try:
                answer = APIHandlers._maybe_improve_core_explanation(answer, query, similar_docs, verification_result.judgment)
            except Exception:
                pass

            # 4-γ. 추가 설명 비거나 빈약할 때 보강
            try:
                if verification_result.judgment in ("올바른 의료 정보", "올바르지 않은 정보"):
                    import re as _re
                    m_add = _re.search(r"📚\s*추가\s*의학\s*정보\s*:\s*([\s\S]*)$", answer)
                    current_add = (m_add.group(1).strip() if m_add else '')
                    if not current_add or len(current_add) < 20 or '관련된 의학 정보를 찾을 수 없습니다' in current_add:
                        better_add = await APIHandlers._generate_additional_medical_info(query, similar_docs)
                        if better_add and better_add != '관련된 의학 정보를 찾을 수 없습니다.':
                            if m_add:
                                answer = answer[:m_add.start(1)] + better_add + answer[m_add.end(1):]
                            else:
                                answer += f"\n\n📚 추가 의학 정보:\n{better_add}"
            except Exception:
                pass
            
            sources = verification_result.evidence_sources
            logger.info(f"API 응답에서 sources 개수: {len(sources)}")
            
            # 5. 섹션 구조 생성 (프론트 직접 사용)
            # 정책: 올바른/올바르지 않은 → 핵심/근거/추가 표시, 판단하기 어려움/민간요법 → 근거만
            core_section = ""
            reason_section = verification_result.reason or ""
            additional_section = ""
            try:
                if verification_result.judgment in ("올바른 의료 정보", "올바르지 않은 정보"):
                    # 핵심 설명: 기존 answer에서 추출(라벨 기반) 또는 RAG 생성 폴백
                    import re as _re
                    m_core = _re.search(r"핵심\s*설명\s*\n([\s\S]*?)(?:\n\n|\n💡|\n판단\s*근거|$)", answer)
                    core_section = (m_core.group(1).strip() if m_core else "")
                    # 지나치게 일반 문구면 RAG 기반 생성으로 대체 시도
                    def _is_generic(txt: str) -> bool:
                        t = (txt or '').strip()
                        return (not t) or len(t) < 15 or t.endswith('확인되었습니다.') or ('정보가 확인되었습니다' in t)
                    if _is_generic(core_section):
                        try:
                            # 동기 컨텍스트: 간단 후보 추출로 대체
                            import re as _re2
                            q_tokens = APIHandlers._extract_core_keywords(query)
                            q_low = query.lower()
                            topic_hints = set()
                            if any(t in q_low for t in ['뎅기','dengue']):
                                topic_hints.update(['뎅기','dengue','이집트숲모기','흰줄숲모기','aedes','모기','바이러스'])
                            if any(t in q_low for t in ['a형간염','a형 간염','a형감염','hepatitis a','hav','간염']):
                                topic_hints.update(['a형간염','a형 간염','hepatitis a','hav','간염','바이러스'])
                            diagnostic_block = ['미노전이효소','ast','got','빌리루빈','혈청','수치','황달','검사','igm','igg','항체','pcr','rt-pcr','유전자','검체','혈액','뇌척수액','serology','항원']
                            other_disease = ['b형간염','hepatitis b','hbv','c형간염','hepatitis c','hcv']
                            cands = []
                            for doc in (similar_docs or [])[:2]:
                                content = (doc.get('content') or '')
                                for s in _re2.split(r"[\.!?]\s+", content):
                                    s = s.strip()
                                    if not s:
                                        continue
                                    s_low = s.lower()
                                    if topic_hints and not any(h in s_low for h in topic_hints):
                                        continue
                                    if any(k in s_low for k in diagnostic_block) or any(k in s_low for k in other_disease):
                                        continue
                                    s_tokens = set(_re2.findall(r"[가-힣A-Za-z0-9]+", s_low))
                                    overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
                                    if overlap >= 0.25 and 20 <= len(s) <= 220:
                                        cands.append((overlap, s))
                            if cands:
                                cands.sort(key=lambda x: x[0], reverse=True)
                                new_core = cands[0][1]
                                if new_core and new_core[-1] not in '.!?':
                                    new_core += '.'
                                core_section = new_core
                        except Exception:
                            pass
                    # 추가 설명: answer에서 블록 추출(있으면) 또는 비었으면 위 생성값 사용
                    import re as _re3
                    m_add = _re3.search(r"📚\s*추가\s*의학\s*정보\s*:\s*([\s\S]*)$", answer)
                    additional_section = (m_add.group(1).strip() if m_add else "")
            except Exception:
                pass

            sections = {
                "core": core_section if verification_result.judgment in ("올바른 의료 정보", "올바르지 않은 정보") else "",
                "reason": reason_section,
                "additional": additional_section if verification_result.judgment in ("올바른 의료 정보", "올바르지 않은 정보") else ""
            }
            
            # 6. 시스템 정보 추가
            system_info = model_manager.get_system_info()
            system_info.update(database_manager.get_database_info())

            response = {
                "answer": answer,
                "sources": sources,
                "sections": sections,
                "verification_details": {
                    "confidence_score": verification_result.confidence_score,
                    "risk_level": verification_result.risk_level,
                    "is_accurate": verification_result.is_accurate,
                    "warnings": verification_result.warnings,
                    "matched_keywords": getattr(verification_result, "matched_keywords", []),
                    "triggered_rules": getattr(verification_result, "triggered_rules", []),
                    "judgment": getattr(verification_result, "judgment", ""),
                    "reason": getattr(verification_result, "reason", "")
                },
                "system_info": system_info
            }
            
            logger.info("쿼리 처리 완료")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"쿼리 처리 중 오류: {e}")
            raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

    @staticmethod
    async def _generate_additional_medical_info(query: str, context_docs: list) -> str:
        """RAG 기반 추가 의학 정보 생성 - 동적 필터링 강화"""
        
        try:
            if not context_docs:
                return "관련된 의학 정보를 찾을 수 없습니다."
            
            # 1. 사용자 질문의 핵심 키워드 추출
            query_keywords = APIHandlers._extract_core_keywords(query)
            query_lower = query.lower()
            
            # 추가: 의도/주제별 키워드 세트 (동적 필터 강화)
            complication_keywords = ['합병증', '기앵-바레증후군', '급성 신부전', '담낭염', '췌장염', '혈관염', '관절염']
            # 질환 토큰: 쿼리에서 동적으로 확장 (간염/뎅기 등)
            base_hep = ['a형간염', '에이형간염', 'a 형 간염', 'a형 간염', '간염', 'hepatitis a']
            dengue_terms = ['뎅기', 'dengue']
            hepatitis_keywords = base_hep + ([] if not any(t in query_lower for t in dengue_terms) else dengue_terms)
            other_disease_keywords = [
                '전염성 단핵구증', '편도염', '인두염', '연쇄상구균', '요관암', '성분화이상',
                # A형간염 맥락에서 배제할 B/C형 간염 관련 용어
                'b형간염', 'b 형 간염', 'b형 간염', 'b형 감염', 'hbv', 'hepatitis b',
                'c형간염', 'c 형 간염', 'c형 간염', 'c형 감염', 'hcv', 'hepatitis c'
            ]
            diagnostic_keywords = [
                '미노전이효소', 'ast', 'got', '빌리루빈', '혈청', '수치', '황달', '검사',
                # 질환 일반화: 검사/검체/분자진단/항체 용어는 추가 설명에서 배제
                'igm', 'igg', '항체', 'pcr', 'rt-pcr', '유전자', '검체', '혈액', '뇌척수액', 'serology', '항원'
            ]
            # 치료 관련 용어(허위 권고 방지): 부정적 맥락(없음/권장X/효과X) 없이 등장 시 제외
            treatment_keywords = ['항생제', '항바이러스', '항비타민', '항비타민제', '특효약', '특효 치료제', '치료제']
            negative_cues = ['없', '아님', '권장되지 않', '비권장', '효과 없', '근거 없']
            # A형간염 맥락에서 명백히 잘못된 주장 필터
            false_claim_patterns = [
                r'전염되\s*지\s*않',
                r'백신\s*은?\s*필요\s*없',
                r'특효.*치료제.*있'
            ]
            
            # 2. 문서 내용에서 관련성 높은 부분만 필터링
            relevant_content = []
            for doc in context_docs[:2]:
                content = doc.get('content', '')
                content_lower = content.lower()
                
                # 사용자 질문과 직접적으로 관련된 문장만 추출
                sentences = content.split('.')
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_lower = sentence.lower()
                    
                    # 핵심 키워드가 포함된 문장만 선택
                    keyword_matches = sum(1 for kw in query_keywords if kw in sentence_lower)
                    
                    # 무관한 키워드가 포함된 문장 제외
                    has_irrelevant = any(kw in sentence_lower for kw in diagnostic_keywords)
                    has_other_disease = any(kw.lower() in sentence_lower for kw in other_disease_keywords)
                    
                    # 관련성 점수 계산
                    relevance_score = keyword_matches / max(1, len(query_keywords))

                    # 의도 기반 강화 조건: 합병증 질의일 때
                    if '합병증' in query_lower:
                        # 간염/헤파타이티스 맥락 확인
                        has_hepatitis = any(kw in sentence_lower for kw in hepatitis_keywords)
                        # 합병증 관련 키워드 포함 확인
                        has_complication = any(kw in sentence_lower for kw in complication_keywords)
                        # 쿼리 문장 자체를 그대로 반복한 문장 제외 (거의 동일한 경우)
                        q_tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", query_lower))
                        s_tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", sentence_lower))
                        overlap_ratio = len(q_tokens & s_tokens) / max(1, len(q_tokens))

                        if (not has_complication and '합병증' not in sentence_lower) or not has_hepatitis:
                            continue
                        if has_other_disease:
                            continue
                        if has_irrelevant:
                            continue
                        # 너무 동일한 문장(거의 복붙)은 제외하여 부연설명 다양성 확보
                        if overlap_ratio > 0.95:
                            continue
                    
                    # 치료 관련 용어가 부정 맥락 없이 포함되면 제외
                    has_treatment = any(kw in sentence_lower for kw in treatment_keywords)
                    negated_treatment = any(cue in sentence_lower for cue in negative_cues)
                    if has_treatment and not negated_treatment:
                        continue
                    # 명백한 허위 주장 패턴 제외
                    if any(re.search(pat, sentence_lower) for pat in false_claim_patterns):
                        continue
                    # 불완전/생략부호 문장 제외
                    if ('...' in sentence) or ('…' in sentence) or re.search(r'[\.]{2,}$', sentence):
                        continue
                    if re.search(r'(으로|로|며|고|등)$', sentence):
                        continue
                    # 관련성이 높고 무관한 키워드가 없는 문장만 포함
                    if relevance_score >= 0.3 and not has_irrelevant and not has_other_disease:
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    # 문장 수를 2개로 제한하여 집중도 향상
                    clipped = relevant_sentences[:2]
                    relevant_content.append('. '.join(clipped) + '.')
            
            # 3. 관련 내용이 없으면 기본 응답
            if not relevant_content:
                    return "관련된 의학 정보를 찾을 수 없습니다."
            
            # 4. 검증된 근거 문장 기반 안전 요약 반환 (문장 단위, 다문장 허용)
            if relevant_content:
                raw_text = ' '.join(relevant_content)
                # 리스트 기호/과도한 공백 제거
                cleaned = re.sub(r"[•\-\u2022]+", ' ', raw_text)
                # 과도/비정상 공백 정규화 (중간에 띄어쓰기 깨짐 방지)
                cleaned = re.sub(r"\s{2,}", ' ', cleaned)
                cleaned = re.sub(r"(\S)\s{1}(\S)", r"\1 \2", cleaned)  # 단일 공백만 유지
                cleaned = cleaned.strip()
                # 일본어/태국어 등 비한글 스크립트 제거
                cleaned = re.sub(r"[\u3040-\u30FF\u31F0-\u31FF\uFF65-\uFF9F\u0E00-\u0E7F]+", '', cleaned)

                # 문장 단위 분할 후 3~4문장으로 재조합
                sentences = [s.strip() for s in re.split(r"[\.?!]\s+", cleaned) if s.strip()]
                picked = []
                for s in sentences:
                    if 10 <= len(s) <= 220:
                        picked.append(s)
                    if len(picked) >= 4:
                        break
                if not picked:
                    picked = sentences[:2]
                # 문장 종결 보정 및 과도 공백 제거
                result_text = '. '.join([s.rstrip(' .') for s in picked])
                # 중복 접두어(예: "합병증 A형간염의 합병증") 제거
                result_text = re.sub(r'^합병증\s+(?=A형간염의\s+합병증)', '', result_text)
                if result_text and not re.search(r"[\.!?]$", result_text):
                    result_text += '.'
                # 어절 이상 공백 교정(있 습 니다 등) 및 과도 공백 제거
                result_text = re.sub(r'있\s*습\s*니\s*다', '있습니다', result_text)
                result_text = re.sub(r'없\s*습\s*니\s*다', '없습니다', result_text)
                result_text = re.sub(r'됩\s*니\s*다', '됩니다', result_text)
                result_text = re.sub(r"\s{2,}", ' ', result_text).strip()
                return result_text
                
        except Exception as e:
            logger.warning(f"추가 의학 정보 생성 실패: {e}")
            return "관련된 의학 정보를 찾을 수 없습니다."

    @staticmethod
    async def _generate_core_explanation(query: str, context_docs: list) -> str:
        """RAG 기반 핵심 설명 생성 (1-2문장, 직접적/의학적 설명)"""
        try:
            if not context_docs:
                return ""
            # 문장 후보 추출: 질의 키워드와의 겹침 + A형간염 맥락 강제 + 진단/타질환 배제
            q_tokens = APIHandlers._extract_core_keywords(query)
            hepatitis_a_terms = ['a형간염', 'a형 간염', 'a형감염', 'hepatitis a', 'hav', '간염']
            diagnostic_keywords = ['미노전이효소', 'ast', 'got', '빌리루빈', '혈청', '수치', '황달', '검사']
            other_disease_keywords = ['b형간염','b 형 간염','b형 간염','hbv','hepatitis b','c형간염','c 형 간염','c형 간염','hcv','hepatitis c',
                                      '전염성 단핵구증','편도염','인두염','연쇄상구균','요관암','성분화이상']

            import re as _re
            candidates = []
            for doc in context_docs[:2]:
                content = (doc.get('content') or '')
                for s in _re.split(r"[\.!?]\s+", content):
                    s = s.strip()
                    if not s:
                        continue
                    s_lower = s.lower()
                    has_hep_a = any(t in s_lower for t in hepatitis_a_terms)
                    has_diag = any(k in s_lower for k in diagnostic_keywords)
                    has_other = any(k in s_lower for k in other_disease_keywords)
                    if not has_hep_a or has_diag or has_other:
                        continue
                    s_tokens = set(_re.findall(r"[가-힣A-Za-z0-9]+", s_lower))
                    overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
                    if overlap >= 0.25 and 20 <= len(s) <= 220:
                        candidates.append((overlap, s))
            if not candidates:
                return ""
            # 겹침 높은 순으로 1-2문장 선택
            candidates.sort(key=lambda x: x[0], reverse=True)
            picked = [candidates[0][1]]
            if len(candidates) > 1 and candidates[1][0] >= 0.3:
                picked.append(candidates[1][1])
            core = '. '.join([p.rstrip(' .') for p in picked])
            if core and core[-1] not in '.!?':
                core += '.'
            core = re.sub(r'있\s*습\s*니\s*다', '있습니다', core)
            core = re.sub(r'없\s*습\s*니\s*다', '없습니다', core)
            core = re.sub(r'됩\s*니\s*다', '됩니다', core)
            return core
        except Exception as e:
            logger.warning(f"핵심 설명 생성 실패: {e}")
            return ""

    @staticmethod
    def _maybe_improve_core_explanation(answer: str, query: str, context_docs: list, judgment: str) -> str:
        """핵심 설명이 일반적이거나 비어있으면 RAG로 대체 생성하여 치환"""
        try:
            if judgment not in ("올바른 의료 정보", "올바르지 않은 정보"):
                return answer
            import re as _re
            # 핵심 블록 추출: '핵심 설명' 다음 블록(💡 또는 빈줄 전까지)
            m = _re.search(r"핵심\s*설명\s*\n([\s\S]*?)(?:\n\n|\n💡|\n판단\s*근거|$)", answer)
            current_core = (m.group(1).strip() if m else '')
            def is_generic(txt: str) -> bool:
                t = (txt or '').strip()
                if not t:
                    return True
                if len(t) < 15:
                    return True
                return t.endswith('확인되었습니다.') or ('정보가 확인되었습니다' in t)
            if is_generic(current_core):
                # 동기 컨텍스트에서 core 생성 호출은 async 불가 → 간단 후보 생성으로 대체
                # 비동기 함수는 상위에서 호출해야 하지만, 여기서는 보수적으로 기존 문서에서 추출
                try:
                    import re as _re2
                    q_tokens = APIHandlers._extract_core_keywords(query)
                    hepatitis_a_terms = ['a형간염','a형 간염','a형감염','hepatitis a','hav','간염']
                    diagnostic_keywords = ['미노전이효소','ast','got','빌리루빈','혈청','수치','황달','검사']
                    other_disease_keywords = ['b형간염','b 형 간염','b형 간염','hbv','hepatitis b','c형간염','c 형 간염','c형 간염','hcv','hepatitis c']
                    cands = []
                    for doc in (context_docs or [])[:2]:
                        content = (doc.get('content') or '')
                        for s in _re2.split(r"[\.!?]\s+", content):
                            s = s.strip()
                            if not s:
                                continue
                            s_low = s.lower()
                            if not any(t in s_low for t in hepatitis_a_terms):
                                continue
                            if any(k in s_low for k in diagnostic_keywords) or any(k in s_low for k in other_disease_keywords):
                                continue
                            s_tokens = set(_re2.findall(r"[가-힣A-Za-z0-9]+", s_low))
                            overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
                            if overlap >= 0.25 and 20 <= len(s) <= 220:
                                cands.append((overlap, s))
                    if cands:
                        cands.sort(key=lambda x: x[0], reverse=True)
                        new_core = cands[0][1]
                        if new_core and new_core[-1] not in '.!?':
                            new_core += '.'
                        # 치환
                        if m:
                            answer = answer[:m.start(1)] + new_core + answer[m.end(1):]
                        else:
                            # 핵심 블록이 비정상일 때 안전 삽입
                            answer = re.sub(r"핵심\s*설명\s*\n", f"핵심 설명\n{new_core}\n\n", answer)
                except Exception:
                    pass
            return answer
        except Exception:
            return answer

    @staticmethod
    def _extract_core_keywords(text: str) -> Set[str]:
        """핵심 키워드 추출"""
        tokens = set()
        for tok in re.findall(r"[가-힣A-Za-z0-9]+", text.lower()):
            if len(tok) >= 2:
                tokens.add(tok)
        return tokens

    @staticmethod
    def _is_response_relevant(query: str, response: str) -> bool:
        """동적 응답 관련성 체크"""
        query_lower = query.lower()
        response_lower = response.lower()
        
        # 질문과 응답의 키워드 추출
        query_tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", query_lower))
        response_tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", response_lower))
        
        # 키워드 겹침률 계산
        overlap = len(query_tokens & response_tokens)
        overlap_ratio = overlap / max(1, len(query_tokens))
        
        # 최소 80% 이상 겹쳐야 관련성 있다고 판단
        return overlap_ratio >= 0.8

    @staticmethod
    def _extract_preserve_terms(text: str) -> list:
        """질문 내에서 반드시 보존해야 할 표기(약어/유형)를 추출합니다."""
        terms = set()
        try:
            allowed = getattr(config.verification, 'ALLOWED_MEDICAL_ABBREVIATIONS', []) or []
            for abbr in allowed:
                if abbr and abbr in text:
                    terms.add(abbr)
        except Exception:
            pass
        # 일반 패턴: A/B/C/...형
        for m in re.findall(r"\b([A-Za-z])형\b", text):
            terms.add(f"{m.upper()}형")
        return sorted(terms)


# API 핸들러 인스턴스
api_handlers = APIHandlers()
