# Define scenarios
def scenario(function_name: str) -> list[dict[str, any]]:
    """
    Function to return the scenario for the given function.

    :param function_name: name of the function
    :type function_name: str
    :return: scenario for the given function
    :rtype: list[dict[str, any]]
    """
    if function_name == "route_question":
        return [
            # Vector store routing
            {
                "chain_name": "route_question_vectorstore",
                "state_name": "route_question",
                "expected_output": "vectorstore",
            },
            # Web search routing
            {
                "chain_name": "route_question_websearch",
                "state_name": "route_question",
                "expected_output": "search",
            },
        ]
    elif function_name == "decide_to_generate":
        return [
            # Confidence in retrieved documents
            {
                "state_name": "decide_to_generate_confidence",
                "expected_output": "correct",
            },
            # No confidence in retrieved documents
            {
                "state_name": "decide_to_generate_no_confidence",
                "expected_output": "not_correct",
            }
        ]
    elif function_name == "grade_generation":
        return [
            # Max iterations
            {
                "max_iterations": True,
                "state_name": "grade_generation_max_iters",
                "expected_output": "stop",
            },
            # Answer is not grounded in documents
            {
                "state_name": "grade_generation_lower_iters",
                "hallucination_chain_name": "hallucination_grader_hallucinate",
                "expected_output": "regen",
            },
            # Answer is grounded in documents and factual to the question
            {
                "state_name": "grade_generation_lower_iters",
                "hallucination_chain_name": "hallucination_grader_no_hallucinate",
                "answer_chain_name": "answer_factual",
                "expected_output": "useful",
            },
            # Answer is grounded in documents but not factual to the question
            {
                "state_name": "grade_generation_lower_iters",
                "hallucination_chain_name": "hallucination_grader_no_hallucinate",
                "answer_chain_name": "answer_not_factual",
                "expected_output": "not_useful",
            },
        ]