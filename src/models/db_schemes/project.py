from pydantic import BaseModel, Field, validator
from typing import Optional
from bson import ObjectId

class Project(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    project_id: str= Field(...,min_length=1)

    @validator('project_id')
    def project_id_validator(cls, v):
        if not v.isalnum():
            raise ValueError('Project ID must be alphanumeric')
        return v
    
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"} 

    @classmethod
    def get_indexes(cls):
        return [
            {
                'key': [
                    ("project_id", 1)
                ], 
                'unique': True,
                'name': 'project_id_index_1'
                
                
        }
        ]
       
    
