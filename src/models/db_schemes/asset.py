from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId
from datetime import datetime

class Asset(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    asset_project_id:  Optional[ObjectId] = Field(None, alias="asset_project_id")
    asset_type: str= Field(..., min_length=1)
    asset_name: str= Field(..., min_length=1)
    asset_size: int= Field(ge=0,default= None)
    asset_config: dict= Field(default= None)
    asset_pushed_at:datetime= Field(default= datetime.utcnow)


    class Config:
        arbitrary_types_allowed = True


    @classmethod
    def get_indexes(cls):
        return [
            {
                'key': [
                    ("asset_project_id", 1)
                ], 
                'unique': False ,
                'name': 'asset_project_id_index_1'
            
            },
              {
                'key': [
                    ("asset_project_id", 1),
                    ("asset_name", 1)

                ], 
                'unique': True ,
                'name': 'asset_project_id_name_index_1'
            
            },

        ]