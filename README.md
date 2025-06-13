# RAGs_Langchain  
TThis project was created to help me learn how Retrieval-Augmented Generation (RAG) works and how to build a RAG system.  
I used LangChain along with a transformer model (via the pipeline API).  
The workflow begins by loading a local PDF file, then converting its content into embeddings. These embeddings are stored locally and retrieved later using the Maximal Marginal Relevance (MMR) method.  
