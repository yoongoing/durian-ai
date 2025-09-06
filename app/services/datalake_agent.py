from typing import Any, Literal

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import get_model_name, LLMs
from langchain_teddynote import logging
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.query import AgentState


# === 환경 설정 ===
# .env 파일에서 환경 변수 로드
load_dotenv(override=True)

# 프로젝트 이름 및 버전 설정
logging.langsmith("Durian-AI", "0.1.0")

# 모델 및 데이터베이스 초기화
MODEL_NAME = get_model_name(LLMs.GPT4o)
db = SQLDatabase.from_uri(settings.DATABASE_URL)

# SQLDatabaseToolkit 생성
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=settings.OPENAI_API_KEY)
)

# 툴 목록 가져오기
tools = toolkit.get_tools()

# 테이블 목록 도구
get_list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    
# 테이블 스키마 도구
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")



# === 도구 관련 함수 ===
# Query 실행 도구
@tool
def db_query_tool(query: str) -> str:
    """
    Run SQL queries against a database and return results
    Returns an error message if the query is incorrect
    If an error is returned, rewrite the query, check, and retry
    """
    # 쿼리 실행
    result = db.run_no_throw(query)

    # 오류: 결과가 없으면 오류 메시지 반환
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    # 정상: 쿼리 실행 결과 반환
    return str(result)

# Query 체크 도구
def query_check(state: AgentState) -> AIMessage:
    query_check_system = """You are a SQL expert with a strong attention to detail.
    Double check the SQLite query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

    You will call the appropriate tool to execute the query after running this check."""

    query_check_prompt = ChatPromptTemplate.from_messages(
        [("system", query_check_system), ("placeholder", "{messages}")]
    )

    query_check_chain = query_check_prompt | ChatOpenAI(
        model=MODEL_NAME, temperature=0
    ).bind_tools([db_query_tool], tool_choice="db_query_tool")

    return query_check_chain

# 오류 처리 함수
def handle_tool_error(state) -> dict:
    # 오류 정보 조회
    error = state.get("error")
    # 도구 정보 조회
    tool_calls = state["messages"][-1].tool_calls
    # ToolMessage 로 래핑 후 반환
    return {
        "messages": [
            ToolMessage(
                content=f"Here is the error: {repr(error)}\n\nPlease fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# 오류를 처리하고 에이전트에 오류를 전달하기 위한 ToolNode 생성
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    # 오류 발생 시 대체 동작을 정의하여 ToolNode에 추가
    
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )



# === 노드 관련 함수 ===
def first_tool_call(state: AgentState) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "initial_tool_call_abc123",
                    }
                ],
            )
        ]
    }

# 쿼리의 정확성을 모델로 점검하기 위한 함수 정의
def model_check_query(state: AgentState) -> dict[str, list[AIMessage]]:
    """
    Use this tool to check that your query is correct before you run it
    """
    return {"messages": [query_check(state).invoke({"messages": [state["messages"][-1]]})]}

# 최종 상태를 나타내는 도구 설명
class SubmitFinalAnswer(BaseModel):
    """쿼리 결과를 기반으로 사용자에게 최종 답변 제출"""
    final_answer: str = Field(..., description="The final answer to the user")

# 조건부 에지 정의
def should_continue(state: AgentState) -> Literal[END, "correct_query", "query_gen"]:
    messages = state["messages"]

    last_message = messages[-1]
    if last_message.content.startswith("Answer:"):
        return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"


# === 그래프 빌드 & 캐싱 ===
def query_gen(state: AgentState):
    # 질문과 스키마를 기반으로 쿼리를 생성하기 위한 모델 노드 추가
    QUERY_GEN_INSTRUCTION = """You are a SQL expert with a strong attention to detail.

    You can define SQL queries, analyze queries results and interpretate query results to response an answer.
    Read the messages bellow and identify the user question, table schemas, query statement and query result, or error if they exist.

    1. If there's not any query result that make sense to answer the question, create a syntactically correct SQLite query to answer the user question. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    2. If you create a query, response ONLY the query statement. For example, "SELECT id, name FROM pets;"
    3. If a query was already executed, but there was an error. Response with the same error message you found. For example: "Error: Pets table doesn't exist"
    4. If a query was already executed successfully interpretate the response and answer the question following this pattern: Answer: <<question answer>>. For example: "Answer: There three cats registered as adopted"
    """

    query_gen_prompt = ChatPromptTemplate.from_messages(
        [("system", QUERY_GEN_INSTRUCTION), ("placeholder", "{messages}")]
    )
    query_gen_chain = query_gen_prompt | ChatOpenAI(model=MODEL_NAME, temperature=0).bind_tools(
    [SubmitFinalAnswer])

    return query_gen_chain


# 쿼리 생성 노드 정의
def query_gen_node(state: AgentState):
    message = query_gen(state).invoke(state)

    # LLM이 잘못된 도구를 호출할 경우 오류 메시지를 반환
    tool_messages = []
    message.pretty_print()
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}


_GRAPH = None

def build_graph():
    workflow = StateGraph(AgentState)
    
    # 첫 번째 도구 호출 노드 추가
    workflow.add_node("first_tool_call", first_tool_call)

    # 첫 번째 두 도구를 위한 노드 추가
    workflow.add_node(
        "list_tables_tool", create_tool_node_with_fallback([get_list_tables_tool])
    )
    workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))

    # 질문과 사용 가능한 테이블을 기반으로 관련 테이블을 선택하는 모델 노드 추가
    model_get_schema = ChatOpenAI(model=MODEL_NAME, temperature=0).bind_tools(
        [get_schema_tool]
    )
    workflow.add_node(
        "model_get_schema",
        lambda state: {
            "messages": [model_get_schema.invoke(state["messages"])],
        },
    )

    # 쿼리 생성 노드 추가
    workflow.add_node("query_gen", query_gen_node)

    # 쿼리를 실행하기 전에 모델로 점검하는 노드 추가
    workflow.add_node("correct_query", model_check_query)

    # 쿼리를 실행하기 위한 노드 추가
    workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

    # 노드 간의 엣지 지정
    workflow.add_edge(START, "first_tool_call")
    workflow.add_edge("first_tool_call", "list_tables_tool")
    workflow.add_edge("list_tables_tool", "model_get_schema")
    workflow.add_edge("model_get_schema", "get_schema_tool")
    workflow.add_edge("get_schema_tool", "query_gen")
    workflow.add_conditional_edges(
        "query_gen",
        should_continue,
    )
    workflow.add_edge("correct_query", "execute_query")
    workflow.add_edge("execute_query", "query_gen")

    return workflow.compile(checkpointer=MemorySaver())



def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH