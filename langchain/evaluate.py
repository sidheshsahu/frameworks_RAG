import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset         
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (                            
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
)
from ragas import RunConfig

from core.doc_store import create_index_1
from core.llm_call import llm_1
from core.prompt_template import template_1

load_dotenv()

questions = [
    "What is the main objective of the blockchain course proposal?",
    "Which programming language is used for writing smart contracts in the course?",
    "Name two consensus algorithms discussed in the course.",
    "How does blockchain help in AI integration?",
    "What practical project is suggested in the course?",
    "Which tools are recommended for blockchain development in the course?",
    "What are the hardware requirements for this course?",
    "How is the course evaluation structured?",
    "Which departments can benefit from this blockchain course?",
    "What future opportunities can arise from this course proposal?"
]

ground_truths = [
    "The main objective is to provide theoretical and practical knowledge of blockchain systems, smart contracts, and decentralized applications.",
    "Solidity is used for writing smart contracts.",
    "Proof of Work and Proof of Stake are two consensus algorithms discussed in the course.",
    "Blockchain helps AI integration by securing AI model sharing and maintaining training data provenance.",
    "A Blockchain-based Secure Voting System is suggested as a practical group project.",
    "The course recommends tools such as Metamask, Solidity, Truffle Suite, and Ganache.",
    "The course requires systems with at least 8 GB RAM and GPU-enabled setups for blockchain simulation.",
    "The evaluation consists of theory, practical work, projects, assignments, quizzes, and attendance.",
    "Computer Science, Information Technology, AI & Data Science, and Electronics & Telecommunication departments can benefit from the course.",
    "The course can lead to industry collaborations, internships, research opportunities, and the establishment of a Blockchain Research Cell."
]

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(embedder)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chunks_docs(docs):
    return [doc.page_content for doc in docs]

def build_retriever(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    texts = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(documents)
    vectorstore = PineconeVectorStore.from_documents(
        texts, embedder, index_name=create_index_1()
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

retriever = build_retriever("example.pdf")

parser = StrOutputParser()

answer_chain = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    | template_1()    
    | llm_1()
    | parser
)

dataset_list = []

for query, reference in zip(questions, ground_truths):
    retrieved_docs = retriever.invoke(query)
    retrieved_contexts = [doc.page_content for doc in retrieved_docs]

    response = answer_chain.invoke(query)

    dataset_list.append({
        "user_input": query,
        "retrieved_contexts": retrieved_contexts,
        "response": response,
        "reference": reference
    })

evaluation_dataset = EvaluationDataset.from_list(dataset_list)  

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        Faithfulness(),       
        LLMContextRecall(),   
        FactualCorrectness(),  
        ResponseRelevancy(),   
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=RunConfig(max_workers=1, timeout=120) 
)

score_df = result.to_pandas()
score_df.to_csv("EvaluationScores.csv", encoding="utf-8", index=False)
print(result)
print(score_df)