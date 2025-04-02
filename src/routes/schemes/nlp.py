from pydantic import BaseModel  
from typing import Optional

class PushRequest(BaseModel):
    file_id: Optional[str]=None
    chunk_size: Optional[int] = 100
    overlap_size: Optional[int] = 20
    do_reset: Optional[int] = 0


class SearchRequest(BaseModel):
    text:str
    limit: Optional[int] = 10
