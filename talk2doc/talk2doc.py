from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough  
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.prompts import ChatPromptTemplate  
from langchain.text_splitter import CharacterTextSplitter 


import tempfile
import os
import subprocess

import argparse
import ollama
from datetime import datetime

SUMMARIZE_PROMT="Summarize the provided text in as few words as possible. Use a brief style with short replies."

def _summarize(doc_splits, model_name):
    res = []
    for d in doc_splits:
        a = ollama.chat(model=model_name, messages=[{"role": "user", "content": d.page_content},
                                                    {"role":"system", "content": SUMMARIZE_PROMT}])
        res.append(a["message"]["content"])
    return res


def _get_ollama_models():
    models=[m["name"] for m in ollama.list()["models"]]
    return " \n".join(models)

def talk_to_my_doc(model_name, pdf_files, chunk_size=300, chunk_overlap=25, top_k=6, embed_model="nomic-embed-text"):
    """
    Load PDF documents into a vectorstore and enable interactive question answering.
    
    Args:
        model_name (str): Name of the LLM model to use
        pdf_files (list[str]): List of paths to PDF files
    """
    # Initialize local LLM model instance with specified name
    model_local = ChatOllama(model=model_name)

    # Load and split each PDF document into pages
    pages = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        pages.extend(loader.load_and_split())

    # Split text from each page into chunks (for embedding purposes)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)

    # Create vectorstore instance with specified documents and embedding function
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="science_vii",
        embedding=embeddings.ollama.OllamaEmbeddings(model=embed_model),
        # persist_directory="./science_vii_chromadb"
    )

    # Get retriever instance from vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={'k': top_k})

    # Create chat prompt template with specified format
    rag_template = """Answer the question based only on the following context:
        {context}
        Question:{question}.

        After the answer, please give me 3 more questions that you can answer from this context.
        """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    # Create chat pipeline with specified components
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | model_local
        | StrOutputParser()
    )

    issummarize = False
    while True:
        print("\n\n")
        question = input("Ask a question (/bye to quit, /print to print, /p_summ for summary) :")
        if "/bye" in question:
            break

        if "/print" in question:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tempf= os.path.join(tempfile.gettempdir(),"talk2doc_{}.txt".format(timestamp))
            with open(tempf, "w") as fout:
                fout.write(response)
            p=subprocess.Popen(["open",tempf])
        elif "/p_summ" in question:
            if not issummarize:
                res = _summarize(doc_splits, model_name)
                response = "\n".join(res)
                print(response)
                issummarize = response
            else:
                response = issummarize
                print(response)
            
        else:
            # Invoke chat pipeline with user input and print response
            response = rag_chain.invoke(question)
            print(response)


def main():
    parser = argparse.ArgumentParser(description="Ask questions to documents in a set of PDFs.")
    parser.add_argument('model_name', help='The name of the LLM model to use (e.g., "mistral", "gemma" Available Models: '+_get_ollama_models())
    parser.add_argument('-p', '--pdf_files', nargs='+', help='List of paths to the PDF files')
    parser.add_argument('-s', '--chunk-size', type=int, default=500, help='Chunck Size. Default: %(default)s')
    parser.add_argument('-o', '--chunk-overlap', type=int, default=50, help='Chunk Overlap. Default: %(default)s')
    parser.add_argument('-k', '--top-k', type=int, default=6, help='Top K docs to return. Default: %(default)s')
    parser.add_argument('-em', '--embed-model', type=str, default="nomic-embed-text", help='Embedding Model to use Default: %(default)s')
    args = parser.parse_args()

    talk_to_my_doc(args.model_name, args.pdf_files, chunk_size=args.chunk_size, 
                   chunk_overlap=args.chunk_overlap, top_k=args.top_k, embed_model=args.embed_model)


if __name__ == "__main__":
    main()
