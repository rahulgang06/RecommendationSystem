from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class IndexingData:
    """Class for indexing data from PDF documents"""

    def __init__(self, data_path, db_faiss_path):
        """Initialize IndexingData.

        Args:
            data_path : Path to the directory containing PDF documents
            db_faiss_path : Path to save the FAISS vector store
        """
        self.data_path = data_path
        self.db_faiss_path = db_faiss_path

    def load_documents(self):
        """Load documents from the specified directory"""

        loader = DirectoryLoader(self.data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        return loader.load()

    def split_text(self, documents):
        """Split text from documents into chunks."""

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents(documents)

    def get_embeddings(self, texts):
        """Get embeddings for the given texts."""
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
        return embeddings.embed_documents(texts)


    def create_vector_db(self):
        """Create and save a FAISS vector store."""
        documents = self.load_documents()
        texts = self.split_text(documents)
        
        # Extract page content (text) from each Document object
        texts = [doc.page_content for doc in texts]
        
        embeddings = self.get_embeddings(texts)
        
        # Create FAISS vector store using texts and embeddings
        db = FAISS.from_texts(texts, embeddings)
        db.save_local(self.db_faiss_path)




if __name__ == "__main__":
    # Path to the directory containing PDF files
    data_path = 'data/'
    # Path to save the FAISS vector store
    db_faiss_path = 'vectorstore/db_faiss'
    system = IndexingData(data_path, db_faiss_path)
    system.create_vector_db()




