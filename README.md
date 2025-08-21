# AI 의료정보 검증 백엔드 서버

## 📋 프로젝트 개요

이 프로젝트는 **RAG (Retrieval-Augmented Generation)** 기술을 활용한 AI 기반 의료·약품 정보 검증 시스템의 백엔드 서버입니다. FastAPI를 기반으로 구축되었으며, ChromaDB 벡터 데이터베이스와 Sentence Transformers를 활용하여 의료 정보의 정확성을 실시간으로 검증합니다.

### 🎯 주요 목적
- 의료 정보의 정확성 실시간 검증
- RAG 기반 신뢰할 수 있는 검증 결과 제공
- 고성능 API 서비스 제공
- 확장 가능한 모듈식 아키텍처 구현

## 🏗️ 프로젝트 구조

```
backend/
├── app.py                    # FastAPI 메인 애플리케이션
├── config.py                 # 설정 관리 모듈
├── model_manager.py          # AI 모델 로딩 및 관리
├── database_manager.py       # ChromaDB 연결 및 관리
├── api_handlers.py           # API 요청 처리 핸들러
├── medical_verification.py   # 의료 검증 핵심 로직
├── requirements.txt          # Python 의존성
├── my_chroma_db/             # ChromaDB 데이터 저장소
├── __pycache__/             # Python 캐시 파일
└── README.md                 # 이 파일
```

## 🚀 주요 기능

### 1. 의료 정보 검증
- **실시간 검증**: 입력된 의료 정보의 정확성 즉시 검증
- **신뢰도 점수**: 0-100% 범위의 정량적 신뢰도 제공
- **상세 분석**: 검증 근거 및 설명 자동 생성
- **출처 제공**: 참고된 의료 자료 자동 인용

### 2. RAG (Retrieval-Augmented Generation) 시스템
- **벡터 검색**: ChromaDB를 활용한 의미적 검색
- **문서 검색**: 관련 의료 자료 자동 검색 및 참조
- **임베딩 생성**: Sentence Transformers를 이용한 텍스트 임베딩
- **컨텍스트 강화**: 검색된 문서를 기반으로 한 정확한 답변 생성

### 3. API 서비스
- **RESTful API**: 표준 HTTP 메서드를 활용한 API 설계
- **CORS 지원**: 프론트엔드와의 원활한 연동
- **비동기 처리**: 고성능 비동기 요청 처리
- **자동 문서화**: Swagger/OpenAPI 자동 문서 생성

### 4. 모듈식 아키텍처
- **설정 관리**: 중앙화된 설정 시스템
- **모델 관리**: AI 모델의 효율적인 로딩 및 관리
- **데이터베이스 관리**: ChromaDB 연결 및 쿼리 최적화
- **API 핸들러**: 요청 처리 로직 분리

## 🛠️ 기술 스택

### Web Framework
- **FastAPI 0.104.1** - 고성능 비동기 웹 프레임워크
- **Uvicorn 0.24.0** - ASGI 서버

### AI/ML
- **Sentence Transformers 2.2.2** - 텍스트 임베딩 모델
- **PyTorch ≥2.0.0** - 딥러닝 프레임워크
- **ChromaDB 0.4.18** - 벡터 데이터베이스

### Data Processing
- **Pydantic 2.9.2** - 데이터 검증 및 직렬화
- **NumPy ≥1.24.0** - 수치 계산 라이브러리

### HTTP Client
- **Aiohttp 3.9.5** - 비동기 HTTP 클라이언트
- **Requests 2.32.3** - 동기 HTTP 클라이언트

## 📦 설치 및 실행

### 사전 요구사항
- Python 3.8 이상
- pip 20.0 이상
- 최소 4GB RAM (모델 로딩용)
- 인터넷 연결 (모델 다운로드용)

### 1. 프로젝트 클론
```bash
git clone <repository-url>
cd proto/backend
```

### 2. 가상환경 생성 (권장)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 서버 실행

#### 개발 모드
```bash
python app.py
```

#### Uvicorn 직접 사용
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

#### 프로덕션 모드
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. 서버 접속
- **API 문서**: http://localhost:8000/docs
- **대안 문서**: http://localhost:8000/redoc
- **서버 상태**: http://localhost:8000/

## 🔧 설정

### 환경 변수
프로젝트 루트에 `.env` 파일을 생성하여 설정:

```env
# 서버 설정
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]

# ChromaDB 설정
CHROMA_DB_PATH=./my_chroma_db
COLLECTION_NAME=medical_documents

# AI 모델 설정
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEVICE=cpu  # 또는 cuda (GPU 사용 시)

# API 설정
API_TIMEOUT=30
MAX_QUERY_LENGTH=1000
```

### ChromaDB 설정
- **데이터베이스 경로**: `./my_chroma_db/`
- **임베딩 모델**: `sentence-transformers/all-MiniLM-L6-v2`
- **컬렉션 이름**: `medical_documents`
- **차원**: 384 (모델에 따라 자동 설정)

## 📊 API 엔드포인트

### 1. 서버 상태 확인
```
GET /
```
**응답 예시:**
```json
{
  "message": "AI 의료정보 검증 서버가 실행 중입니다! (RAG 기반)",
  "status": "running",
  "version": "2.0.0"
}
```

### 2. 의료 정보 검증
```
POST /verify
```

**Request Body:**
```json
{
  "query": "검증할 의료 정보 텍스트"
}
```

**Response Body:**
```json
{
  "answer": "검증 결과 요약",
  "sources": [
    {
      "title": "참고 문서 제목",
      "content": "관련 내용",
      "similarity": 0.85
    }
  ],
  "verification_details": {
    "confidence": 85.5,
    "judgment": "정확한 정보",
    "reasoning": "판단 근거",
    "core_explanation": "핵심 설명"
  },
  "system_info": {
    "documents_found": 3,
    "processing_time": 1.2,
    "model_used": "all-MiniLM-L6-v2"
  }
}
```

### 3. 시스템 정보
```
GET /system-info
```
**응답 예시:**
```json
{
  "model_status": "loaded",
  "database_status": "connected",
  "collection_count": 1500,
  "memory_usage": "2.1GB",
  "uptime": "2h 30m"
}
```

### 4. 건강 상태 확인
```
GET /health
```
**응답 예시:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-21T10:30:00Z",
  "services": {
    "model": "ok",
    "database": "ok",
    "api": "ok"
  }
}
```

## 🏗️ 아키텍처 상세

### 1. 설정 관리 (config.py)
- **중앙화된 설정**: 모든 설정을 한 곳에서 관리
- **환경별 설정**: 개발/프로덕션 환경별 설정 분리
- **타입 안전성**: Pydantic을 통한 설정 검증

### 2. 모델 관리 (model_manager.py)
- **지연 로딩**: 필요할 때만 모델 로딩
- **메모리 최적화**: 효율적인 메모리 사용
- **모델 캐싱**: 로딩된 모델 재사용

### 3. 데이터베이스 관리 (database_manager.py)
- **연결 풀링**: 효율적인 데이터베이스 연결 관리
- **쿼리 최적화**: 벡터 검색 성능 최적화
- **에러 처리**: 데이터베이스 오류 복구

### 4. API 핸들러 (api_handlers.py)
- **요청 검증**: 입력 데이터 유효성 검사
- **응답 포맷팅**: 일관된 응답 형식
- **에러 핸들링**: 상세한 오류 메시지

### 5. 의료 검증 (medical_verification.py)
- **RAG 파이프라인**: 검색-생성 파이프라인 구현
- **신뢰도 계산**: 정량적 신뢰도 점수 계산
- **결과 분석**: 상세한 검증 결과 생성

## 🔍 사용 예시

### Python 클라이언트 예시
```python
import requests
import json

# 검증 요청
url = "http://localhost:8000/verify"
data = {
    "query": "아스피린은 혈액을 묽게 만들어 혈전을 예방합니다."
}

response = requests.post(url, json=data)
result = response.json()

print(f"신뢰도: {result['verification_details']['confidence']}%")
print(f"판단: {result['verification_details']['judgment']}")
print(f"설명: {result['verification_details']['core_explanation']}")
```

### cURL 예시
```bash
curl -X POST "http://localhost:8000/verify" \
     -H "Content-Type: application/json" \
     -d '{"query": "아스피린은 혈액을 묽게 만들어 혈전을 예방합니다."}'
```

## 🐛 문제 해결

### 일반적인 문제

#### 1. 모델 로딩 실패
```bash
# 메모리 부족 시
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# CPU 사용 강제
export CUDA_VISIBLE_DEVICES=""
```

#### 2. ChromaDB 연결 오류
```bash
# 데이터베이스 재생성
rm -rf my_chroma_db/
python -c "from database_manager import database_manager; database_manager.connect()"
```

#### 3. 포트 충돌
```bash
# 다른 포트 사용
uvicorn app:app --port 8001
```

#### 4. 의존성 설치 오류
```bash
# pip 업그레이드
pip install --upgrade pip

# 캐시 삭제 후 재설치
pip cache purge
pip install -r requirements.txt
```

### 로깅
- **로그 레벨**: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- **로그 파일**: 콘솔 출력 (파일 로깅은 향후 추가 예정)
- **로그 형식**: `시간 - 모듈 - 레벨 - 메시지`

## 📈 성능 최적화

### 현재 적용된 최적화
- **비동기 처리**: FastAPI의 비동기 특성 활용
- **모델 캐싱**: 로딩된 모델 재사용
- **벡터 인덱싱**: ChromaDB의 효율적인 벡터 검색
- **메모리 관리**: 효율적인 메모리 사용

### 추가 최적화 계획
- **Redis 캐싱**: 자주 사용되는 쿼리 결과 캐싱
- **배치 처리**: 대량 요청 처리 최적화
- **GPU 가속**: CUDA 지원으로 처리 속도 향상
- **로드 밸런싱**: 다중 인스턴스 배포

## 🔒 보안

### 현재 보안 조치
- **CORS 설정**: 허용된 도메인만 접근 가능
- **입력 검증**: Pydantic을 통한 데이터 검증
- **에러 처리**: 민감한 정보 노출 방지
- **타임아웃**: 요청 타임아웃 설정

### 보안 모범 사례
- **HTTPS 사용**: 프로덕션 환경에서 HTTPS 필수
- **API 키 인증**: 향후 API 키 기반 인증 추가 예정
- **Rate Limiting**: 요청 제한 기능 추가 예정
- **로그 보안**: 민감한 정보 로깅 방지

## 🧪 테스트

### 단위 테스트
```bash
# 테스트 실행 (향후 추가 예정)
python -m pytest tests/
```

### API 테스트
```bash
# 서버 상태 확인
curl http://localhost:8000/

# 검증 API 테스트
curl -X POST "http://localhost:8000/verify" \
     -H "Content-Type: application/json" \
     -d '{"query": "테스트 쿼리"}'
```

### 성능 테스트
```bash
# 부하 테스트 (향후 추가 예정)
python -m pytest tests/test_performance.py
```

## 📊 모니터링

### 현재 모니터링
- **서버 상태**: `/health` 엔드포인트
- **시스템 정보**: `/system-info` 엔드포인트
- **로그 모니터링**: 실시간 로그 출력

### 추가 모니터링 계획
- **메트릭 수집**: Prometheus/Grafana 연동
- **알림 시스템**: 오류 발생 시 알림
- **성능 대시보드**: 실시간 성능 모니터링

## 🤝 기여하기

### 개발 환경 설정
1. 프로젝트 포크
2. 로컬에 클론
3. 가상환경 생성
4. 의존성 설치
5. 개발 서버 실행

### 코딩 스타일
- **PEP 8** 준수
- **Type Hints** 사용
- **Docstring** 작성
- **Black** 포맷터 사용

### 테스트
- 새로운 기능에 대한 테스트 작성
- 기존 테스트 통과 확인
- 테스트 커버리지 유지

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

### 이슈 리포트
- GitHub Issues를 통한 버그 리포트
- 기능 요청 및 개선 제안

### 문의사항
- 프로젝트 관련 문의: [이메일 주소]
- 기술 지원: [기술 지원 채널]

---

**버전**: 2.0.0  
**최종 업데이트**: 2025년 8월 21일  
**개발자**: AI 의료정보 검증 팀
