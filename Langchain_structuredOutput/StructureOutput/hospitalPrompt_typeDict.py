from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated,Optional,Literal

model=ChatOllama(model="mistral-nemo")

class HospitalAdmissionSystem(TypedDict):
    name:str 
    age:int 
    weight_kg:float
    blood_type:Literal['A+','B+','O+','A-']
    ward_type:Literal['ICU' ,'General' ,'Emergency' ,'Private']
    condition_severity:Literal['mild' , 'moderate' , 'critical']
    admission_type:Literal['emergency', 'planned']
    diagnosis_notes:Annotated[str,"Tell the notes for diagonsis"]
    pre_existing_conditions:Annotated[list[str],"give list of pre-existing diesases"]
    estimated_stay_days:Annotated[int,"No of days estimated for stay"]
    has_insurance:Annotated[Optional[bool],"Check if a person has insurance if not no need to mention"]
    has_drug_allergy:Annotated[Optional[bool],"Check if a person has any drug allergy if not no need to mention"]
    next_of_kin_notified:Annotated[Optional[bool],"Check if a kin has been notified if not no need to mention"]


prompt="Patient Michael Brown, 45 years old, was admitted on an emergency basis today for chest pain. He is a diabetic and has a history of hypertension. His blood type is O positive. The attending doctor has categorized his condition as critical. He is currently admitted in the ICU ward. Patient is covered under insurance. His weight is 82 kg. No known drug allergies. Next of kin has been notified. Estimated stay is 5 days."


result=model.with_structured_output(HospitalAdmissionSystem).invoke(prompt)

print(result)