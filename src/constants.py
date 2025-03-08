"""Module defining constants to be used over the project."""

# Logger file and output folder
LOG_FILE_DIR = "logs"  # Name of the logs folder
LOG_FILE_NAME = "app.log"  # Name of the log file

# Embedding model
EMBEDDING_MODEL = "thenlper/gte-base" # 768-dimensional embedding model with 512-dimensional context window
RERANKER_MODEL = "BAAI/bge-reranker-large" # cross-encoder to be used for reranking

# Search types
MMR = {
    "search_type": "mmr", # use Maximal Marginal Relevance to retrieve the most relevant documents
    "search_kwargs": {
        "k": 10, # number of documents to retrieve
        "fetch_k": 25, # number of documents to retrieve before MMR post processing
        "lambda_mult": 0.6, # weight between diversity and relevance (the higher the lower the diversity and the higher the relevance)
    },
}
SIMILARITY = {
    "search_type": "similarity", # use Maximal Marginal Relevance to retrieve the most relevant documents
    "search_kwargs": {
        "k": 10, # number of documents to retrieve
    },
}

TOP_K = 5 # number of documents to pass to the Graders (this is after either MMR or Similarity search and reranking)

# Nodes
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Max generations iterations
MAX_ITERATIONS = 2 # allows a maxium of 3 generations

# KB topics
TOPICS = [
    "agent",
    "agent memory",
    "prompt engineering",
    "adversarial attacks",
]

# LLMs templates

# Template for the Self-reflection -> Grading if the answer is factual to the posed question
ANSWER_GRADER_TEMPLATE = """
You are an evaluator grading and assessing if a passage addresses and resolves a question.
Give only a 'yes' or 'no' binary score.
Output 'yes' if the passage addresses and resolves the question, and 'no' when the passage does not address and resolve the question.

User question: What is linear regression?
Passage: Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables.
In simple terms, it's a way to predict the value of the dependent variable based on the values of the independent variables.
Here's a breakdown of how it works:
Dependent Variable (Target): The variable you are trying to predict or explain.
Independent Variable(s): The variables used to predict or explain the dependent variable.
In the simplest case of simple linear regression, you have one independent variable. The relationship between the independent variable (let's call it X) and the dependent variable (let's call it Y) is represented by a straight line.
Answer: yes

User question: Who is the author of the book "The Great Gatsby"?
Passage: The author of The Divine Comedy is Dante Alighieri, an Italian poet, writer, and philosopher.
The Divine Comedy, written in the early 14th century, is one of the most important works in world literature.
It is an epic poem that describes Dante's journey through the three realms of the dead: Inferno (Hell), Purgatorio (Purgatory), and Paradiso (Paradise).
The work is a profound allegory of the soul's journey toward God, and it is considered one of the greatest achievements in Italian literature and Western thought.
Answer: no

User question: Can you give me a short bio about Andre Agassi?
Passage: Andre Agassi is a former American professional tennis player, widely regarded as one of the greatest players in the history of the sport.
Born on April 29, 1970, in Las Vegas, Nevada, Agassi turned professional in 1986 at the age of 16.
Known for his powerful baseline play and charismatic personality, he won a total of 8 Grand Slam singles titles, including the Australian Open (four times), French Open, Wimbledon, and the US Open.
Agassi was the first male player in the Open Era to win all four Grand Slam tournaments, completing a career Grand Slam.
His career also included an Olympic gold medal in 1996.
Agassi was ranked world No. 1 in singles for a total of 60 weeks during his career.
Off the court, Agassi is known for his philanthropic work, especially through the Andre Agassi Foundation for Education, which helps at-risk children in Las Vegas.
He retired from professional tennis in 2006 and has since remained involved in various charitable causes and business ventures.
His autobiography, "Open," published in 2009, became a best-seller and provided a candid look at his life and career.
Answer: yes

User question: Can you give me a short explanation of non-parametric regression?
Passage: Logistic regression is a statistical method used for binary classification tasks, where the goal is to predict one of two possible outcomes (often coded as 0 or 1).
It's similar to linear regression but differs in the way it models the relationship between the dependent and independent variables, especially since the output is a probability that is constrained between 0 and 1.
Answer: no

User question: {question}
Passage: {generation}
Answer:
"""

# Template for the Self-reflection -> Grading if the answer is factual to the retrieved documents/context
HALLUCINATION_GRADER_TEMPLATE = """
You are an evaluator grading and assessing if a passage is grounded and supported by the given set of documents.
Give only a 'yes' or 'no' binary score.
Output 'yes' if the passage is supported by the sets of documents, and 'no' when the passage is not supported by the sets of documents.

Documents:
The Eiffel Tower was constructed in 1889 for the World's Fair in Paris.
It is a popular tourist attraction and one of the most recognizable structures in the world.
Passage: The Eiffel Tower was built in 1889 and is an iconic symbol of France, attracting millions of visitors each year.
Answer: yes

Documents:
The Great Wall of China is one of the largest man-made structures in the world, stretching over 13,000 miles.
The Wall was built over several dynasties to protect Chinese states from invasions.
Passage: The Great Wall of China was constructed to be a tourist attraction and make China more popular around the globe.
Answer: no

Documents:
The moon orbits Earth every 27.3 days.
It is the fifth-largest natural satellite in the solar system.
Passage: Mars is a large planet with a thin atmosphere and two moons.
Answer: no

Documents:
The theory of relativity was developed by Albert Einstein in the early 20th century.
It revolutionized the understanding of space, time, and gravity.
Passage: Albert Einstein's work on the theory of relativity reshaped scientific views on the nature of space and time.
Answer: yes

Documents: {documents}
Passage: {generation}
Answer:
"""

# Template for the Self-correction -> Grading if the question is grounded into the retrieved documents/context
RETRIEVAL_GRADER_TEMPLATE = """
You are a grader assessing the relevance of retrieved documents to a user question.
If the document contains keywords or semantic meaning related to the question, grade it as relevant.
Give a  binary score of 'yes' or 'no' to indicate whether the document is relevant or not to the user question.

Retrieved documents:
The Eiffel Tower was constructed in 1889 for the World's Fair in Paris.
It is a popular tourist attraction and one of the most recognizable structures in the world.
User question: When was the Eiffel Tower built?
Answer: yes

Retrieved documents:
The Great Wall of China is one of the largest man-made structures in the world, stretching over 13,000 miles.
The Wall was built over several dynasties to protect Chinese states from invasions.
User question: What is the Statue of Liberty?
Answer: no

Retrieved documents:
The moon orbits Earth every 27.3 days.
It is the fifth-largest natural satellite in the solar system.
User question: What are the main characteristics of Mars?
Answer: no

Retrieved documents:
The theory of relativity was developed by Albert Einstein in the early 20th century.
It revolutionized the understanding of space, time, and gravity.
User questin: Who developed the theory of relativity?
Answer: yes

Retrieved document: {document}
User question: {question}
Answer:
"""

# Template for question router -> it routes to either vector store or web search based on the question topic
QUESTION_ROUTER_TEMPLATE = """
You are an expert at routing a user question to a vectorstore or websearch.
The vectorstore contains documents that related to:
{topics}
Use the vectorstore for questions on these topics, for all else use websearch.

User question: What is ReAct prompt?
Answer: vectorstore

User question: Which is the current president of USA?
Answer: websearch

User question: What is the current temperature in Celsius degrees in London?
Answer: websearch

User question: What is token manipulation in LLMs adversarial attacks?
Answer: vectorstore

User question: {question}
Answer:
"""

# Template for function caller Agent -> it calls the right tool based on the question
FUNCTION_CALLER_TEMPLATE = """
Today is {today}.
You are an helpful assistant in answering user questions.
Use the tools to answer the question and stick to the tools output.
You have access to the following tools: {tools}

Question: {query}
{agent_scratchpad}

Answer:
"""

# Template for the Generation of the answer
GENERATION_TEMPLATE = """
Use the following pieces of retrieved context to answer the question at the end.
If you do not know the answer, just say that you do not know, do not try to make up an answer.

<context>
{context}
</context>

Question: {question}
Answer:
"""