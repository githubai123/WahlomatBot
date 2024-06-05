#
from crewai_tools import SerperDevTool, PDFSearchTool
from PyPDF2 import PdfReader

class HelperTools:
    def __init__(self,model_provider_rag_="ollama",model_base_="http://localhost:11434",llm_model_name_="crewai-llama3", embedding_model_="nomic-embed-text", **kwargs):
        self.llm_model_name_ = llm_model_name_
        self.model_provider_rag_ = model_provider_rag_
        self.embedding_model_ = embedding_model_
        self.model_provider_base_ = model_base_
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_tool_pdf_rag(self, pdf_name):
        return PDFSearchTool(
        pdf=pdf_name,
        config=dict(
            llm=dict(
                provider=self.model_provider_rag_,
                config=dict(
                    model=self.llm_model_name_,
                    base_url=self.model_provider_base_,  
                ),
            ),
            embedder=dict(
                provider="ollama",
                config=dict(
                    model=self.embedding_model_,
                    base_url=self.model_provider_base_,  
                ),
            ),
        )
    )
    
    
    def get_tool_pdf_rag_party(self, party):
        return self.get_tool_pdf_rag(f'data/{party}.pdf')
    

    def get_serper_search_tool(self):
        return SerperDevTool()

    
    def read_pdf(self,file_path):
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            content = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_content = page.extract_text()
                # Strip non UTF8 characters
                page_content = page_content.encode('utf-8', 'ignore').decode('utf-8')
                content += page_content
        return content



