"""
AI 기반 의료·약품 정보 검증 백엔드 서버
리팩토링된 모듈식 구조 (RAG 기반)
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 로컬 모듈 임포트
from config import config
from model_manager import model_manager
from database_manager import database_manager
from api_handlers import api_handlers

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, config.server.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic 모델
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    verification_details: dict = {}
    system_info: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 모델 로딩/해제"""
    logger.info("🚀 AI 의료정보 검증 서버 시작 중...")
    
    # 모델 로딩
    model_success, model_msg = await model_manager.load_models()
    logger.info(f"모델 로딩 결과: {model_msg}")
    
    # 데이터베이스 연결
    db_success, db_msg = database_manager.connect()
    logger.info(f"데이터베이스 연결 결과: {db_msg}")
    
    if model_success or db_success:
        logger.info("✅ 서버 준비 완료!")
    else:
        logger.warning("⚠️ 일부 기능이 제한될 수 있습니다.")
    
    yield
    
    # 정리 작업
    logger.info("서버 종료 중...")
    database_manager.disconnect()
    logger.info("👋 서버 종료 완료")


# FastAPI 앱 생성
app = FastAPI(
    title="AI 의료정보 검증 API",
    description="ChromaDB와 RAG를 활용한 의료정보 검증 시스템 (RAG 기반)",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "message": "AI 의료정보 검증 서버가 실행 중입니다! (RAG 기반)",
        "status": "running",
        "version": "2.0.0"
    }


@app.get("/favicon.ico")
async def favicon():
    """파비콘 요청 처리 (404 방지)"""
    return {"message": "No favicon"}


@app.middleware("http")
async def log_requests(request, call_next):
    """요청 로깅 미들웨어"""
    client_ip = request.client.host
    method = request.method
    url = str(request.url)
    
    logger.info(f"🌐 {client_ip} - {method} {url}")
    
    response = await call_next(request)
    
    logger.info(f"✅ {client_ip} - {method} {url} - {response.status_code}")
    
    return response


@app.get("/api/health")
async def health_check():
    """시스템 상태 및 정보 확인"""
    return await api_handlers.health_check()


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """쿼리를 처리하고 의료 정보 검증 결과를 반환합니다."""
    result = await api_handlers.process_query(request.query)
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        verification_details=result["verification_details"],
        system_info=result["system_info"]
    )


if __name__ == "__main__":
    print("🚀 AI 의료정보 검증 백엔드 서버 시작...")
    print(f"- 주소: http://{config.server.HOST}:{config.server.PORT}")
    print(f"- API 문서: http://{config.server.HOST}:{config.server.PORT}/docs")
    print(f"- 상태 확인: http://{config.server.HOST}:{config.server.PORT}/api/health")
    print("\n서버를 중지하려면 Ctrl+C를 누르세요.\n")
    
    import uvicorn
    uvicorn.run(
        app, 
        host=config.server.HOST, 
        port=config.server.PORT, 
        log_level=config.server.LOG_LEVEL.lower(),
        reload=config.server.DEBUG,
        timeout_keep_alive=300,  # 연결 유지 시간 5분 (한국어 모델용)
        timeout_graceful_shutdown=60,  # 종료 대기 시간 1분
        access_log=False  # 성능 향상
    )
