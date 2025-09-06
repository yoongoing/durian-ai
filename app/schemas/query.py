from typing import Optional, List, Dict, Any
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
import operator

# 에이전트의 상태 정의
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class QueryRequest(TypedDict):
    query_text: str

class QueryResponse(TypedDict):
    columns: List[str]
    rows: List[List[Any]]
    metadata: Dict[str, Any]

