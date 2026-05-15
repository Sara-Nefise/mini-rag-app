from pydantic import BaseModel
from typing import Optional

class PushRequest(BaseModel):
    do_reset: Optional[int] = 0

class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 5
    chat_id: Optional[int]= None
    retrieval_profile: Optional[str] = "baseline"
    #: Per-request override for CE vs dense+RRF fusion (0–1). None = use ``RERANKER_FUSION_CE_WEIGHT`` from settings.
    reranker_fusion_ce_weight: Optional[float] = None

