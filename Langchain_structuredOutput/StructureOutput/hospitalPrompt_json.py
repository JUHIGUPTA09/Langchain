from langchain_ollama import ChatOllama
from typing import Optional,Literal
from pydantic import BaseModel,Field

model=ChatOllama(model="mistral-nemo")

json_schema={
  "title": "HospitalAdmissionSystem",
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "name of the patient"
    },
    "age": {
      "type": "integer"
    },
    "weight_kg": {
      "type": "number"
    },
    "blood_type": {
      "enum": ["A+", "B+", "O+", "A-"],
      "type": "string"
    },
    "ward_type": {
      "enum": ["ICU", "General", "Emergency", "Private"],
      "type": "string"
    },
    "condition_severity": {
      "enum": ["mild", "moderate", "critical"],
      "type": "string"
    },
    "admission_type": {
      "enum": ["emergency", "planned"],
      "type": "string"
    },
    "diagnosis_notes": {
      "type": "string",
      "description": "Notes for diagnosis"
    },
    "pre_existing_conditions": {
      "type": "array",
      "items": { "type": "string" },
      "description": "give list of pre-existing diseases"
    },
    "estimated_stay_days": {
      "type": "integer",
      "exclusiveMinimum": 0,
      "exclusiveMaximum": 10,
      "description": "No of estimated stay"
    },
    "has_insurance": {
      "anyOf": [{ "type": "boolean" }, { "type": "null" }],
      "default": None,
      "description": "Check if a person has insurance if not no need to mention"
    },
    "has_drug_allergy": {
      "anyOf": [{ "type": "boolean" }, { "type": "null" }],
      "default": None,
      "description": "Check if a person has any drug allergy if not no need to mention"
    },
    "next_of_kin_notified": {
      "anyOf": [{ "type": "boolean" }, { "type": "null" }],
      "default": None,
      "description": "Check if a kin has been notified if not no need to mention"
    }
  },
  "required": [
    "name", "age", "weight_kg", "blood_type", "ward_type",
    "condition_severity", "admission_type", "diagnosis_notes",
    "pre_existing_conditions", "estimated_stay_days"
  ]
}

prompt="Patient Michael Brown, 45 years old, was admitted on an emergency basis today for chest pain. He is a diabetic and has a history of hypertension. His blood type is O positive. The attending doctor has categorized his condition as critical. He is currently admitted in the ICU ward. Patient is covered under insurance. His weight is 82 kg. No known drug allergies. Next of kin has been notified. Estimated stay is 5 days."


result=model.with_structured_output(json_schema).invoke(prompt)

print(result)