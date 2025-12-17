import pytest
from pydantic import BaseModel
from loom.client import LoomClient

# --- Define a Schema for the Test ---
class HardwareSpecs(BaseModel):
    cpu_name: str
    core_count: int
    base_clock_ghz: float
    supports_pcie_5: bool

@pytest.mark.integration
def test_structured_data_extraction():
    """
    Verifies that the 120B model can extract specific fields from unstructured text
    and return a valid Python object (not just a string).
    """
    client = LoomClient()
    
    raw_text = """
    The AMD Ryzen Threadripper PRO 3945W is a beast. 
    It features 12 cores running at a base speed of 4.0GHz. 
    While it has many PCIe lanes, it is a Gen4 part, so no Gen5 support.
    """

    # Execute Extraction
    result = client.generate_structured(
        system_prompt="You are an expert hardware engineer. Extract the specifications.",
        user_prompt=raw_text,
        response_model=HardwareSpecs
    )

    # Verify the object is fully parsed
    assert isinstance(result, HardwareSpecs)
    assert "3945W" in result.cpu_name
    assert result.core_count == 12
    assert result.base_clock_ghz == 4.0
    assert result.supports_pcie_5 is False # It should infer this from "no Gen5 support"

    print(f"\nExtracted: {result.model_dump_json(indent=2)}")