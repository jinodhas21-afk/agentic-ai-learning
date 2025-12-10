from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv


load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini",
api_key=os.getenv("OPENAI_API_KEY"))

print("="*60)
print("LCEL (LangChain Expression Language) - The Pipe Operator |")
print("=" * 60)


#Example_1: Simple Chain
print("\nExample 1: Simple Chain (Prompt | LLM | Output Parser")
print("-"*60)

template_1="Exaplain {topic} in one sentence"
prompt_1=PromptTemplate(
    template=template_1,
    input_variables=["topic"]
)

chain_1= prompt_1|llm|StrOutputParser()

result_1 = chain_1.invoke({"topic":"Quantum Computing"})
print(f"Topic: Quantum Computing")
print(f"Result: {result_1}\n")

#Example 2: Multi- step Chain
print("EXAMPLE 2: Multi-Step Chain (Step 1 → Step 2)")
print("-" * 60)

template_2a="Exaplain {concept} in 2 sentences"
prompt_2a = PromptTemplate(
    template= template_2a,
    input_variables=["concept"]
)


template_2b="Now summarize this in one word:{text}"
prompt_2b = PromptTemplate(
    template= template_2b,
    input_variables=["text"]
)


# Step 1 chain
chain_2a = prompt_2a | llm | StrOutputParser()
result_2a = chain_2a.invoke({"concept":"Neural Networks"})
print(f"Step 1 Result (Explanation):\n{result_2a}\n")

# Step 2 chain (uses output from step 1)
chain_2b = prompt_2b | llm | StrOutputParser()
result_2b = chain_2b.invoke({"text": result_2a})
print(f"Step 2 Result (One word summary): {result_2b}\n")

#Example 3: Three-step chain
print("EXAMPLE 3: Three-Step Chain")
print("-" * 60)


template_3 = """Given this topic: {topic}
Provide:
1. A definition
2. A real-world use case"""

prompt_3 = PromptTemplate(template=template_3, input_variables=["topic"])
chain_3 = prompt_3 | llm | StrOutputParser()

result_3 = chain_3.invoke({"topic":"Machine Learning"})
print(result_3)

print("\n" + "=" * 60)
print("✅ LCEL Chain Examples Completed!")
print("=" * 60)
print("\nKey Takeaway:")
print("LCEL uses the pipe operator | to chain components together")
print("It's like: Input | Transform | Transform | Output")