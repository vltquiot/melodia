import os
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import glob
from langchain_core.documents import Document
import gc  # Garbage collector for memory management

DATA_PATHS = [
    "data/artists_infos/",
    "data/tracks_infos/"
]

CHUNK_SIZE = 500 
CHUNK_OVERLAP = 50
FAISS_INDEX_PATH = "faiss_index"
BATCH_SIZE = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_documents_from_folder(folder_paths):
    documents = []
    
    for folder_path in folder_paths:
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        
        if len(txt_files) == 0:
            print(f"No .txt files found in {folder_path}")
            continue
        
        print(f"Loading documents from {folder_path}...")
        folder_docs = 0
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    if not content.strip():
                        continue
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": os.path.basename(file_path),
                            "folder": folder_path
                        }
                    )
                    documents.append(doc)
                    folder_docs += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {folder_docs} documents from {folder_path}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def create_faiss_index_batch(documents):
    print("\n" + "="*60)
    print("CREATING FAISS INDEX WITH BATCH PROCESSING")
    print("="*60 + "\n")
    
    print("Step 1/4: Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': DEVICE}
    )
    print(f"Embedding model loaded on {DEVICE}")
    
    print(f"\nStep 2/4: Splitting documents into chunks...")
    print(f"  - Chunk size: {CHUNK_SIZE}")
    print(f"  - Chunk overlap: {CHUNK_OVERLAP}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)
    print(f"Created {total_chunks} chunks from {len(documents)} documents")
    
    print(f"\nStep 3/4: Creating embeddings and building FAISS index...")
    print(f"  - Batch size: {BATCH_SIZE} chunks")
    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  - Total batches: {total_batches}")
    print(f"  - This will take approximately {total_batches * 0.5:.1f}-{total_batches * 2:.1f} minutes\n")
    
    vectorstore = None
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_chunks)
        batch_chunks = chunks[i:batch_end]
        batch_num = (i // BATCH_SIZE) + 1
        
        print(f"  [{batch_num}/{total_batches}] Processing chunks {i+1}-{batch_end}...", end=' ', flush=True)
        
        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                print("(Initial index)")
            else:
                batch_vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                vectorstore.merge_from(batch_vectorstore)
                print("Ok")
                
                # Clean up to free memory
                del batch_vectorstore
                gc.collect()
                
        except Exception as e:
            print(f"\nError processing batch {batch_num}: {e}")
            raise
    
    print(f"\nFAISS index created with {vectorstore.index.ntotal} vectors!")
    
    return vectorstore


def save_faiss_index(vectorstore, path):
    print(f"\nStep 4/4: Saving FAISS index to {path}...")
    
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)
    
    print(f"Index saved successfully!")
    print(f"  - Location: {os.path.abspath(path)}")


def verify_index(path):
    print(f"\nVerifying saved index...")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': DEVICE}
        )
        
        vectorstore = FAISS.load_local(
            path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        test_results = vectorstore.similarity_search("test", k=1)
        
        print(f"Index verified successfully!")
        print(f"  - Total vectors: {vectorstore.index.ntotal}")
        print(f"  - Test search returned: {len(test_results)} result(s)")
        
        return True
    except Exception as e:
        print(f"Error verifying index: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MUSIC RAG - FAISS INDEX BUILDER (BATCH MODE)")
    print("="*60 + "\n")

    for folder in DATA_PATHS:
        if not os.path.exists(folder):
            print(f"⚠ Warning: Folder '{folder}' not found!")
    
    try:
        documents = load_documents_from_folder(DATA_PATHS)
        
        if len(documents) == 0:
            print(f"\nERROR: No documents found in {DATA_PATHS}")
            print("Please add .txt files to the data folders")
            exit(1)
        
        vectorstore = create_faiss_index_batch(documents)
        
        save_faiss_index(vectorstore, FAISS_INDEX_PATH)
        
        verify_index(FAISS_INDEX_PATH)
        
        print("\n" + "="*60)
        print("INDEX BUILDING COMPLETE!")
        print("="*60)
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nIndex building failed. Please check the error above.")
        exit(1)