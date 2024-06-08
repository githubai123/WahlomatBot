from AdaptiveRagAgent import AdaptiveRagAgent 


class PoliticianAgent(AdaptiveRagAgent):
    def __init__(self,agent_name_="Poiltician",rag_documents=[],**kwargs):
        print("Creating new agent")
        model_provider_rag_="ollama"
        model_base_="http://localhost:11434"
        llm_model_name_="crewai-llama3:8b"
        embedding_model_="nomic-embed-text"
        super().__init__()
        self.add_pdf(rag_documents)







