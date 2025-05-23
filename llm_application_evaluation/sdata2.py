from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas.testset import TestsetGenerator
import time

load_dotenv()

# Google Gemini Configuration
config = {
    "model": "gemini-1.5-pro",  # or other model IDs
    "temperature": 0.4,
    "max_tokens": None,
    "top_p": 0.8
}

path = r"D:\Sample_Docs_Markdown"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

# Initialize with Google AI Studio
generator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
    model=config["model"],
    temperature=config["temperature"],
    max_tokens=config["max_tokens"],
    top_p=config["top_p"],
))

generator_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Google's text embedding model
    task_type="retrieval_document"  # Optional: specify the task type
))

# Introduce delays to manage rate limits
def rate_limited_call(generator, docs, testset_size):
    try:
        return generator.generate_with_langchain_docs(docs, testset_size=testset_size)
    except Exception as e:
        print(f"Rate limit reached, retrying in 10 seconds: {e}")
        time.sleep(60)  # Retry after delay
        return generator.generate_with_langchain_docs(docs, testset_size=testset_size)

# Generate synthetic dataset
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = rate_limited_call(generator, docs, testset_size=10)

# Convert dataset to Pandas DataFrame
print(dataset.to_pandas())
