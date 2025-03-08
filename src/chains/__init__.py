"""
Package dedicated to all chains used within the Graph Nodes.

The main modules are:
* answer_grader: Chain to assess hallucination in the answer compared to the original question.
* generation: Chain to generate an answer to a question.
* retrieval_grader: Chain to assess relevance of a document to a question.
* hallucination_grader: Chain to assess hallucination in the answer compared to the retrieved documents.
* router: Chain to route the question to the right search (i.e., vector search or web search).
* tool_agent: Chain defining a function calling agent to call tools (i.e., Tavily Search or Wikipedia).
"""