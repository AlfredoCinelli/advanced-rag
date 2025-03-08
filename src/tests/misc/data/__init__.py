# Import packages and modules
from langchain_core.documents import Document


# Define scenarios
def scenario(
    function_name: str,
) -> dict[str, any]:
    if function_name == "format_docs":
        return [
            {
                "documents": [
                    Document(
                        page_content="Agent memory is a type of memory that allows an agent to store information about its environment and use that information to make decisions. It is a type of memory that is used by artificial intelligence systems to store information about the environment and use that information to make decisions."
                    ),
                    Document(
                        page_content="Linear algebra is a branch of mathematics that studies vectors, matrices, and linear transformations. It is used in many areas of mathematics, including geometry, analysis, and probability. Linear algebra is also used in many applications, such as computer graphics, physics, and engineering."
                    ),
                ],
                "expected_output": "Agent memory is a type of memory that allows an agent to store information about its environment and use that information to make decisions. It is a type of memory that is used by artificial intelligence systems to store information about the environment and use that information to make decisions.\n\nLinear algebra is a branch of mathematics that studies vectors, matrices, and linear transformations. It is used in many areas of mathematics, including geometry, analysis, and probability. Linear algebra is also used in many applications, such as computer graphics, physics, and engineering.",
            },
        ]
