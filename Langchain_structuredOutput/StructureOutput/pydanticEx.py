from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str
    age:int = 50
    address:Optional[str] =None
    email:EmailStr
    cgpa:float= Field(gt=0,lt=10,description="cgpa of a student",default=5)

new_std={'name':'juhi','email':'abc@gmail.com'}

std_obj=Student(**new_std)
print(std_obj)