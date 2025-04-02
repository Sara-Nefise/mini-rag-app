from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId

class DataChunk(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    chunk_text: str = Field(..., min_length=1)
    chunk_metadata: dict
    chunk_order: int = Field(..., gt=0)
    chunk_project_id:  Optional[ObjectId] = Field(None, alias="chunk_project_id")
    chunk_asset_id:Optional[ObjectId] = Field(None, alias="chunk_asset_id")
    class Config:
        arbitrary_types_allowed = True


    @classmethod
    def get_indexes(cls):
        return [
            {
                'key': [
                    ("chunk_project_id", 1)
                ], 
                'unique': False ,
                'name': 'chunk_project_id_index_1'
            
                
 }
        ]
    

class RetrievedDocument(BaseModel):
        text:str
        score:float