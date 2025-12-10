from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
llm=ChatOpenAI(model="gpt-4o-mini",
api_key=os.getenv("OPENAI_API_KEY"))

# Example 1: simple template
print("="*50)
print("Example 1: Simple Prompt Template")
print("="*50)


template_1="Explain {concept} in simple terms"
prompt_1= PromptTemplate(
    template= template_1,
    input_variables=["concept"]
)

chain_1= prompt_1|llm|StrOutputParser()
result_1=chain_1.invoke({"concept":"Machine Learning"})
print(f"Question:Explain Machine Learning in simple terms")
print(f"Answer:{result_1}\n")

#Example_2: Multi-variable Template
print("="*50)
print("Example 2:  Multi-variable Template")
print("="*50)

template_2="""You are an expert in{domain}.
Answer this question:{question}
Format: Keep it concise (2-3 sentences)"""

prompt_2= PromptTemplate(
    template=template_2,
    input_variables=["domain","question"]
)

chain_2=prompt_2|llm|StrOutputParser()
result_2=chain_2.invoke({
    "domain":"Artifical Intelligence",
    "question":"What is the future if AI?"
})
print(f"Expert Domain: AI")
print(f"Question:What is thr future of AI?")
print(f"Answer: {result_2}\n")

#Example 3: Few -shot Prompting
print("=" * 50)
print("EXAMPLE 3: Few-Shot Prompting")
print("=" * 50)

template_3="""Aswer the question like a technical expert.

Examples:
Q:What is python?
A:Python is a high-level programming language known for readabiity and simplicity.

Q: What is JavaScript?
A: JavaScript is a programming language primarily used for web development.

Now answer this:
Q:{question}
A:"""

prompt_3 = PromptTemplate(
    template=template_3,
    input_variables=["question"]
)

chain_3 = prompt_3 | llm |StrOutputParser()
result_3 = chain_3.invoke({"question": "What is Rust?"})
print(f"Question: What is Rust?")
print(f"Answer: {result_3}\n")

print("✅ All prompt examples completed!")