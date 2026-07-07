from typing_extensions import TypedDict
from typing import Optional

class AgentState(TypedDict):
    probability: float          # Model's raw sigmoid output
    attention_focus: str        # Human-readable summary of attention spread
    notes: str
    wbc: float
    crp: float
    analysis: Optional[str]     # Filled in by analysis_node
    reasoning: Optional[str]    # Filled in by reasoning_node
    report: Optional[str]       # Filled in by report_node