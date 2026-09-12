import json
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from toolsandbox_pipeline.schemas.offline_memory import MemoryReviewOutput
from toolsandbox_pipeline.reproducibility import canonical_sha256

@pytest.mark.parametrize('obj',[
 {'decision':'ADD','reason':'Reusable'},
 {'decision':'SKIP','reason':'Duplicate'},
 {'decision':'MERGE','reason':'Equivalent','target_memory_id':'policy_1'},
])
def test_valid(obj):
 Draft202012Validator(MemoryReviewOutput.model_json_schema()).validate(obj)
 assert MemoryReviewOutput.model_validate_json(json.dumps(obj)).review().decision==obj['decision']

@pytest.mark.parametrize('obj',[
 {'decision':'ADD','reason':'Reusable','target_memory_id':None},
 {'decision':'SKIP','reason':'Duplicate','target_memory_id':None},
 {'decision':'ADD','reason':'Reusable','target_memory_id':'policy_1'},
 {'decision':'MERGE','reason':'Equivalent'},
 {'decision':'MERGE','reason':'Equivalent','target_memory_id':None},
 {'decision':'MERGE','reason':'Equivalent','target_memory_id':''},
 {'decision':'MERGE','reason':'Equivalent','target_memory_id':'has space'},
 {'decision':'ADD','reason':'x'*241},
 {'decision':'ADD','reason':'Reusable','extra':True},
])
def test_wire_rejects(obj):
 assert not Draft202012Validator(MemoryReviewOutput.model_json_schema()).is_valid(obj)

@pytest.mark.parametrize('obj',[
 {'decision':'ADD','reason':'Reusable'},
 {'decision':'SKIP','reason':'Duplicate'},
 {'decision':'MERGE','reason':'Equivalent','target_memory_id':'policy_1'},
])
def test_serialization_roundtrip(obj):
 model=MemoryReviewOutput.model_validate_json(json.dumps(obj))
 assert model.model_dump()==obj
 assert json.loads(model.model_dump_json())==obj
 assert MemoryReviewOutput.model_validate_json(model.model_dump_json())==model
 assert MemoryReviewOutput.model_validate(model.model_dump())==model

def test_merge_null_rejected_before_review():
 with pytest.raises(ValidationError):
  MemoryReviewOutput.model_validate_json('{"decision":"MERGE","target_memory_id":null,"reason":"Equivalent"}')
