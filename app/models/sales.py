from sqlalchemy import Column, Integer, String, Date
from app.db.session import Base

class Sales(Base):
    __tablename__ = "sales"

    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Integer, nullable=False)
    date = Column(String, nullable=False)  # 단순화를 위해 문자열로 저장
