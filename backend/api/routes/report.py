from fastapi import APIRouter, HTTPException
from backend.agent.graph import agent_executor
from backend.agent.utils import summarize_attention

router = APIRouter()

@router.post("/report")
async def generate_report(payload: dict):
    # 1. Run your inference module to extract values first
    # logits, prob, attn_weights = run_inference(payload)
    
    # 2. Map structural values cleanly right into your compiled graph
    initial_state = {
        "probability": 0.84, # Example prob output
        "attention_focus": summarize_attention(attn_weights=None), # Pass attention weights tensor here
        "notes": payload.get("notes", ""),
        "wbc": payload.get("wbc", 0.0),
        "crp": payload.get("crp", 0.0)
    }
    
    # Run the graph execution pipeline
    final_state = agent_executor.invoke(initial_state)
    
    return {"report": final_state["report"]}