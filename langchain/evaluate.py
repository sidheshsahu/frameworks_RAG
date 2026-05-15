import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from core.doc_store import create_index_1
from core.llm_call import llm_1
from core.prompt_template import template_1
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
from datasets import Dataset




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

answers = [
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

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chunks_docs(docs):
    return [doc.page_content for doc in docs]

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)


def rag_pipeline(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PineconeVectorStore.from_documents(
        texts, embedder, index_name=create_index_1()
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    return retriever



def evaluate_rag_pipeline(query,file_path):
    retriever = rag_pipeline(file_path);
    rag_chain = RunnableParallel(
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(format_docs),
            "chunks_context":retriever | RunnableLambda(chunks_docs)
        }
    )

    parser = StrOutputParser()
    output = rag_chain | template_1() | llm_1() | parser
    
    return {
        "answer": output,
        "contexts": rag_chain.chunks_context
    }




answers = []
contexts = []

for query in queries:
    result = rag_pipeline(
        query=query,
        file_path="example.pdf"
    )

    answers.append(result["answer"])
    contexts.append(result["contexts"])


data = {
    "question": queries,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

dataset = Dataset.from_dict(data)



score = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    ],
    llm=evaluator_llm
)



score_df = score.to_pandas()

score_df.to_csv(
    "EvaluationScores.csv",
    encoding="utf-8",
    index=False
)

print(score_df)













