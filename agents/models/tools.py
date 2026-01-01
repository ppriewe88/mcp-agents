import logging
import re
from typing import List, Literal, Optional

from pydantic import BaseModel

from agents.models.client import (
    OpenAIFunction,
    OpenAITool,
    OpenAIToolParameters,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DROP_EMPTY_DEFAULTS_MARKER = "EMPTY"
EmptyDefault = Literal["EMPTY"]


class ToolArg(BaseModel):
    """Contains info how input of a specific mcp tool looks on server."""

    name_on_server: str
    name_for_llm: str
    description_for_llm: str
    type: str = "string"
    required: bool = True
    default: Optional[str | EmptyDefault] = None


class ToolArgsSchema(BaseModel):
    """Type for parameters of mcp tools in openai format.

    Notice: additionalProperties set to False to enforce strict mode of tool calling.
    """

    type: Literal["object"] = "object"
    properties: List[ToolArg]
    additionalProperties: bool = False  # noqa: N815


class ToolSchema(BaseModel):
    """Client-side schema definition for specific mcp tool."""

    server_url: Optional[str] = "http://127.0.0.1:8000/sse"
    name_on_server: str
    name_for_llm: str
    description_for_llm: str
    args_schema: ToolArgsSchema

    def model_post_init(self, __context):
        """Run validation after model initialization."""
        self.validate_schema()
        logger.debug(f"[ToolSchema validation] {self.name_for_llm}: Schema successfully validated")

    def validate_schema(self) -> None:
        """Validates the schema by enforcing all internal consistency rules.

        This method serves as a high-level validator that aggregates all detailed
        validation routines ensuring server/LLM consistency and argument structure
        correctness. It is the central entry point for checking that a ToolSchema
        is complete and safe to use.
        """
        self.validate_required_args()
        self.validate_llm_names()
        return None

    def validate_llm_names(self) -> None:
        """Validates that all LLM-facing names follow OpenAI naming rules.

        OpenAI requires that function names and parameter names match a strict
        character pattern. This method checks the tool name
        and all LLM-visible argument names and descriptions to ensure they contain only allowed
        ASCII characters.

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: If name_for_llm of the tool or any argument contains
                        forbidden characters or violates OpenAI's naming pattern.
        """
        valid_llm_string_pattern = re.compile(r"^[a-zA-Z0-9_\.-]+$")
        # validate tool name
        if not valid_llm_string_pattern.match(self.name_for_llm):
            raise ValueError(
                f"[ToolSchema ERROR] Invalid LLM tool name '{self.name_for_llm}'. "
                f"Allowed characters: letters, digits, '_', '-', '.'"
            )

        # validate argument names
        for arg in self.args_schema.properties:
            if not valid_llm_string_pattern.match(arg.name_for_llm):
                raise ValueError(
                    f"[ToolSchema ERROR] Invalid LLM argument name '{arg.name_for_llm}'. "
                    f"Allowed characters: letters, digits, '_', '-', '.'"
                )

        logger.debug(
            f"[ToolSchema validation] {self.name_for_llm}: All LLM names validated successfully."
        )

    def validate_required_args(self) -> None:
        """Checks the required args and default args for consistency.

        This method derives the effective set of required LLM-visible arguments from
        the per-argument flags 'required' and 'drop_and_inject' and logs them. It
        also enforces that every optional LLM-visible argument (required=False)
        defines a default value, ensuring predictable behavior when the LLM omits
        that argument.
        """
        required_llm = []
        for arg in self.get_args():  # only LLM-visible args
            if arg.required:
                required_llm.append(arg.name_for_llm)
            else:
                # validate: optional args must define a default
                if arg.default is None:
                    raise ValueError(
                        "[ToolSchema ERROR] Optional LLM argument without default detected.\n"
                        f"Tool: {self.name_for_llm}\n"
                        f"Argument: {arg.name_for_llm}\n"
                        "All arguments with required=False must define a default value."
                    )
        logger.debug(
            f"[ToolSchema validation] {self.name_for_llm}: "
            f"Derived required LLM args = {required_llm}"
        )
        return None

    def get_args_schema_for_llm(self) -> dict:
        """Builds the OpenAI/LangChain-compatible argument schema for the LLM.

        Only non-dropped arguments are exposed to the LLM. The 'required' list is
        derived automatically from arguments that are visible to the LLM and have
        required=True.

        Args:
            None

        Returns:
            dict: A JSON-schema dictionary containing 'type', 'properties',
                'required', and 'additionalProperties' suitable for a LangChain
                StructuredTool definition.
        """
        # only non-drop inputs
        active_llm_inputs = [arg for arg in self.args_schema.properties]

        # build properties dict
        properties = {}
        for arg in active_llm_inputs:
            prop = {
                "type": arg.type,
                "description": arg.description_for_llm,
            }
            if arg.default is not None:
                prop["default"] = arg.default
            properties[arg.name_for_llm] = prop

        # derive required values automatically
        required = [arg.name_for_llm for arg in active_llm_inputs if arg.required]

        # build args schema
        args_schema_llm = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        return args_schema_llm

    def get_server_args(self) -> List[ToolArg]:
        """Returns the full list of server-side argument definitions.

        This method provides all ToolArg objects as they appear in the schema,
        including dropped and non-dropped arguments. It is useful for validation,
        logging, and constructing server-side signatures.

        Args:
            None

        Returns:
            List[ToolArg]: A list of all server-side ToolArg definitions.
        """
        return [arg for arg in self.args_schema.properties]

    def get_all_server_arg_names(self) -> List[str]:
        """Returns the list of server-side input names.

        This method extracts all server-visible argument names directly from the
        schema. It is commonly used to verify server-side completeness or to build
        validation logic during tool execution.

        Args:
            None

        Returns:
            List[str]: A list of all server-side argument names.
        """
        return [arg.name_on_server for arg in self.args_schema.properties]

    def get_args(self) -> List[ToolArg]:
        """Returns the args of the tool.

        Args:
            None

        Returns:
            List[ToolArg]: A list of ToolArg objects visible to the LLM.
        """
        return [arg for arg in self.args_schema.properties]
    
    def get_llm_args_names(self) -> List[str]:
        """Returns the names of all arguments visible to the LLM.

        The method extracts only the name_for_llm attributes of arguments that the LLM
        is expected to populate. Dropped arguments are excluded.

        Args:
            None

        Returns:
            List[str]: A list of LLM-visible argument names.
        """
        return [arg.name_for_llm for arg in self.args_schema.properties]

    def get_openai_schema(self) -> OpenAITool:
        """Builds the complete OpenAI tool schema with strict function-calling enabled.

        This method bundles the tool name, description, argument schema,
        and strict mode into an OpenAITool object that can be directly used
        in LangChain or OpenAI function-calling. It ensures that tools follow
        OpenAI's official specification.

        Args:
            None

        Returns:
            OpenAITool: The fully constructed schema for OpenAI/LangChain tool calling.
        """
        openai_schema = OpenAITool(
            function=OpenAIFunction(
                name=self.name_for_llm,
                description=self.description_for_llm,
                parameters=OpenAIToolParameters(**self.get_args_schema_for_llm()),
                strict=True,
            )
        )
        OpenAIToolParameters
        return openai_schema


#################################################################
#################################################################
#################################################################
#################################################################
if __name__ == "__main__":

    test_doodle = ToolSchema(
        name_on_server="blablubb",
        name_for_llm="hi there",
        description_for_llm="Aufrufen, wenn gefragt wird wie der Preis sich ändert bei neuer Fahrleistung",
        args_schema=ToolArgsSchema(
            properties=[
                ToolArg(
                    name_on_server="name",
                    name_for_llm="vorname",
                    description_for_llm="Name des Kunden.",
                    type="string",
                    required=True,
                ),
                ToolArg(
                    name_on_server="age",
                    name_for_llm="alter",
                    description_for_llm="alter des kunden.",
                    type="string",
                    default="20",
                    required=False,
                ),
            ],
            additionalProperties=False,
        ),
    )

    test_doodle.validate_required_args()
    test_doodle.validate_llm_names()
    args_schema = test_doodle.get_args_schema_for_llm()
    print(args_schema)
    test_doodle.get_server_args()
    test_doodle.get_all_server_arg_names()
    test_doodle.get_llm_args_names()
    openai_schema = test_doodle.get_openai_schema()
    print("ENDE")
