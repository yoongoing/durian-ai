from fastapi import FastAPI
from app.routers import datalake
from app.db.session import Base, engine
from app.models.sales import Sales
from app.core.config import settings
from sqlalchemy import select, func

# DB 테이블 생성
Base.metadata.create_all(bind=engine)

# 샘플 데이터 삽입 (SQLite 전용, 비어있을 때만)
if settings.DATABASE_URL.startswith("sqlite"):
    with engine.begin() as conn:
        result = conn.execute(select(func.count()).select_from(Sales)).scalar()
        if result == 0:
            conn.execute(Sales.__table__.insert(), [
                {"id": 1, "amount": 100, "date": "2023-01-01"},
                {"id": 2, "amount": 200, "date": "2023-01-02"},
                {"id": 3, "amount": 300, "date": "2023-01-03"},
            ])

app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(datalake.router)

@app.get("/")
def root():
    return {"message": f"{settings.PROJECT_NAME} is running"}
