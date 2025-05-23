from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ragas.testset import TestsetGenerator

load_dotenv()

path = r"D:\Sample_Docs_Markdown"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

# Filter valid documents
valid_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
if not valid_docs:
    raise ValueError("No valid documents to process!")

# Initialize LLM (Flan-T5)
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator_llm = LangchainLLMWrapper(model)

# Initialize Embeddings (Sentence-BERT)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
generator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# Initialize Testset Generator
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

# Generate test set
dataset = generator.generate_with_langchain_docs(valid_docs, testset_size=10)
print(dataset.to_pandas())
