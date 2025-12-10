"""
Day 1 project: Simple Q&A Chatbot using LangChain
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Initalize LMM
llm= ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

#create Q&A chain
qa_prompt= PromptTemplate(
    template ="""You are a helpful AI assistant for learning agentic AI.

Question:{question}

Please provide al clear, concise answer suitable for someine learning AI.

""",
input_variables=["qquestion"]
)

qa_chain=qa_prompt|llm|StrOutputParser()

#Test questions
test_questions=[
    "What is agentic AI?",
    "What is RAG?",
    "What is LangChain used for?",
    "Explain embeddings in simple terms",
    "What is the ReAct pattern?"
]

print("=" * 70)
print("SIMPLE Q&A CHATBOT - DAY 1 PROJECT")
print("=" * 70)

for i, question in enumerate(test_questions, 1):
    print(f"\n📝 Question {i}: {question}")
    print("-" * 70)
    answer = qa_chain.invoke({"question": question})
    print(f"💡 Answer:\n{answer}")
    print()

print("=" * 70)
print("✅ Day 1 Project Complete!")
print("=" * 70)