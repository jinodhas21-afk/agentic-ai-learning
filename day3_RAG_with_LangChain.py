from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini",
api_key=os.getenv("OPENAI_API_KEY"))


# Load PDF document
loader = PyPDFLoader("SAMPLE HR MANUAL.pdf")
doc=loader.load()

#test splitter
from langchain_classic.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(doc)

from langchain_openai import OpenAIEmbeddings
embedding =OpenAIEmbeddings()

from langchain_community.vectorstores import FAISS
vectorstore= FAISS.from_documents(docs,embedding)


#Retriever

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

# create a prompt template
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("""Use the following pieces of context to answer the question at the end.
If you don't know the answer, say that you don't know.
Context: {context} Question: {question}""")

from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
chain=(
    {"context":RunnableLambda(lambda x: retriever.invoke(x['question'])),
    "question":RunnablePassthrough(),}
    | prompt
    | llm
    | StrOutputParser()
)


result=chain.invoke({"question":"What is the updated leave policy here?"})
print(result)

print(chain.invoke({"question": "What is HR policy?"}))
