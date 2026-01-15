import json
import os
from pydantic import ValidationError
from typing import Dict, Any, Optional
from openai import OpenAI, APIConnectionError, APIStatusError
from pydantic import BaseModel, Field

class LoomResponse(BaseModel):
    """
    A standardized response object for Loom inference requests.

    Attributes:
        content (str): The generated text content from the model.
        token_usage (Dict[str, int]): A dictionary containing 'prompt_tokens',
            'completion_tokens', and 'total_tokens'.
        model_used (str): The specific model alias used for generation.
        finish_reason (str): The reason the generation stopped (e.g., 'stop', 'length').
    """
    content: str
    reasoning: Optional[str] = None
    token_usage: Dict[str, int]
    model_used: str
    finish_reason: str = Field(default="unknown")


class LoomClient:
    """
    A client wrapper for the local Loom (Llama.cpp) inference engine.

    This client abstracts the OpenAI-compatible API provided by llama.cpp,
    enforcing type safety and ensuring inputs meet contract requirements
    before transmission.
    """

    def __init__(
        self, 
        # UPDATED: Defaults for the new Spark infrastructure
        host: str = os.getenv("LOOM_HOST", "http://192.168.1.42:8000/v1"), 
        model_name: str = os.getenv("LOOM_MODEL", "gpt-oss-120b"),
        api_key: str = "EMPTY"
    ) -> None:
        """
        Initialize the LoomClient.

        Args:
            host (str): The full URL to the local inference server API (v1).
                        Defaults to the specific static IP of 'Lenny'.
            model_name (str): The model alias defined in the systemd service.
            api_key (str): The API key (dummy key required by the protocol).
        """
        self.client = OpenAI(base_url=host, api_key=api_key)
        self.model_name = model_name

    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = -1,
        enable_reasoning: bool = True,
    ) -> LoomResponse:
        """
        Generates a text completion from the local Loom engine.

        Pre-conditions:
            - system_prompt must not be empty or whitespace only.
            - user_prompt must not be empty or whitespace only.
            - temperature must be between 0.0 and 2.0.

        Post-conditions:
            - Returns a valid LoomResponse object.
            - The returned content is a string (possibly empty if generation failed completely).
            - Raises APIConnectionError if the server is unreachable.

        Args:
            system_prompt (str): The behavior instructions for the model.
            user_prompt (str): The specific input query to process.
            temperature (float): Controls randomness (0.0 = deterministic).
            max_tokens (int): The limit for generation (-1 for infinity/context limit).

        Returns:
            LoomResponse: Structured response containing the text and metadata.

        Raises:
            ValueError: If pre-conditions regarding prompt content or temperature are violated.
            APIConnectionError: If the Threadripper server cannot be reached.
            APIStatusError: If the server returns a non-200 status code.
        """
        # Validate Pre-conditions
        if not system_prompt.strip():
            raise ValueError("Pre-condition failed: system_prompt cannot be empty.")
        if not user_prompt.strip():
            raise ValueError("Pre-condition failed: user_prompt cannot be empty.")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"Pre-condition failed: temperature {temperature} is out of range (0.0-2.0).")

        try:
            # Execute Request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens if max_tokens > 0 else None,
                extra_body={"enable_reasoning": enable_reasoning}
            )

            # Process Response
            choice = response.choices[0]
            message = choice.message

            # Extract the hidden reasoning field (vLLM specific)
            # The OpenAI library stores unknown fields in 'model_extra' or attributes
            reasoning_content = getattr(message, "reasoning_content", None)
            if not reasoning_content and hasattr(message, "model_extra"):
                reasoning_content = message.model_extra.get("reasoning_content")
            
            # Explicitly extract only the standard fields to avoid Pydantic errors
            # caused by 'None' values in new OpenAI library fields (like token_details).
            usage_stats = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

            # Satisfy Post-conditions via Pydantic validation
            return LoomResponse(
                content=choice.message.content or "",
                reasoning=reasoning_content,
                token_usage=usage_stats,
                model_used=response.model,
                finish_reason=choice.finish_reason
            )

        except APIConnectionError as e:
            # You might want to log this failure specifically in the future
            print(f"Loom Connection Error: Could not reach {self.client.base_url}")
            raise e
        except APIStatusError as e:
            print(f"Loom Status Error: Server returned {e.status_code}")
            raise e
        
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        """
        Generates a structured JSON response enforcing a Pydantic schema.

        Pre-conditions:
            - system_prompt and user_prompt must not be empty.
            - response_model must be a subclass of pydantic.BaseModel.

        Post-conditions:
            - Returns an instance of response_model populated with the LLM's output.
            - Raises ValidationError if the LLM output does not match the schema.

        Args:
            system_prompt (str): The behavior instructions.
            user_prompt (str): The specific input query.
            response_model (type[BaseModel]): The Pydantic class defining the desired JSON structure.
            temperature (float): Low temperature (0.1) recommended for extraction.

        Returns:
            BaseModel: An instance of the provided Pydantic model.
        """
        # 1. Generate the JSON Schema from the Pydantic model
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        # 2. Append instructions to the system prompt
        # Note: Llama.cpp server supports 'response_format={"type": "json_object"}' 
        # which activates generic JSON mode. Forcing the schema in the prompt 
        # is often the most robust 'universal' method for local models.
        guided_system_prompt = (
            f"{system_prompt}\n\n"
            f"You must respond with valid JSON strictly following this schema:\n"
            f"```json\n{schema_str}\n```\n"
            f"Do not add any markdown formatting or chatter."
        )

        try:
            # 3. Call the generic generate method
            # We enforce json_object mode to ensure the model outputs valid JSON syntax
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": guided_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"} 
            )

            content = response.choices[0].message.content or "{}"
            
            # 4. Validate and Parse
            return response_model.model_validate_json(content)

        except ValidationError as e:
            print(f"Loom Validation Error: Model output did not match schema.\nOutput: {content}")
            raise e
        except APIStatusError as e:
            print(f"Loom Status Error: {e.status_code}")
            raise e
