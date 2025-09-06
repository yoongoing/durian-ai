from fastapi import APIRouter
from app.schemas.query import AgentState
from fastapi.responses import FileResponse
from app.services.datalake_agent import get_graph
from app.db.session import SessionLocal
from sqlalchemy import inspect

from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid, invoke_graph, stream_graph
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError


router = APIRouter(prefix="/datalake", tags=["datalake"])


# === POST: 자연어 쿼리 실행 ===
@router.post("/query")
def run_graph(
    message: str, recursive_limit: int = 30, node_names=[], stream: bool = False
):
    app = get_graph()

    # config 설정(재귀 최대 횟수, thread_id)
    config = RunnableConfig(
        recursion_limit=recursive_limit, configurable={"thread_id": random_uuid()}
    )

    # 질문 입력
    inputs = {
        "messages": [HumanMessage(content=message)],
    }

    try:
        if stream:
            # 그래프 실행
            stream_graph(app, inputs, config, node_names=node_names)
        else:
            invoke_graph(app, inputs, config, node_names=node_names)
        output = app.get_state(config).values
        return output
    except GraphRecursionError as recursion_error:
        print(f"GraphRecursionError: {recursion_error}")
        output = app.get_state(config).values
        return output


# === GET: 테이블 목록 ===
@router.get("/tables")
def list_tables():
    with SessionLocal() as db:
        inspector = inspect(db.bind)
        tables = inspector.get_table_names()
    return {"tables": tables}


# === GET: 테이블 스키마 ===
@router.get("/schemas")
def get_schemas():
    with SessionLocal() as db:
        inspector = inspect(db.bind)
        schema_info = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            schema_info[table_name] = [col["name"] for col in columns]
    return {"schemas": schema_info}