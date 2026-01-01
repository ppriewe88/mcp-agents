from agents.models.tools import (
    ToolArg,
    ToolArgsSchema,
    ToolSchema,
)

schema_add = ToolSchema(
    server_url="http://127.0.0.1:8000/sse",
    name_on_server="add_numbers",
    name_for_llm="add",
    description_for_llm="",
    args_schema=ToolArgsSchema(
        properties=[
            ToolArg(
                name_on_server="a",
                name_for_llm="a",
                description_for_llm="erster summand",
                type="string",
            ),
            ToolArg(
                name_on_server="b",
                name_for_llm="b",
                description_for_llm="zweiter summand",
                type="string",
            ),
        ],
        additionalProperties=False,
    ),
)

schema_birthday = ToolSchema(
    server_url="http://127.0.0.1:8001/sse",
    name_on_server="get_birthday_santaclaus",
    name_for_llm="geburtsjahr_weihnachtsmann_ermitteln",
    description_for_llm="liefert das geburtsjahr des weihnachtsmannes",
    args_schema=ToolArgsSchema(
        properties=[
            ToolArg(
                name_on_server="query",
                name_for_llm="anfrage",
                description_for_llm="anfrage wann der weihnachtsmann geboren ist",
                type="string",
            )
        ],
        additionalProperties=False,
    ),
)

schema_more_info_on_santa = ToolSchema(
    server_url="http://127.0.0.1:8000/sse",
    name_on_server="summarize",
    name_for_llm="weitere_infos_weihnachtsmann",
    description_for_llm="""liefert weitere informationen zum weihnachtsmann. 
    Braucht das geburtsjahr des weihnachtsmannes als input""",
    args_schema=ToolArgsSchema(
        properties=[
            ToolArg(
                name_on_server="birth_year",
                name_for_llm="geburtsjahr",
                description_for_llm="geburtsjahr des weihnachtsmanns",
                type="string",
            )
        ],
        additionalProperties=False,
    ),
)