### simple adaptive rag test 
import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain import hub
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_community.utilities import GoogleSerperAPIWrapper
from pprint import pprint
from langchain_community.document_loaders import PyMuPDFLoader

class MyAdaptiveRagAgent():

    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[str]

    def __init__(self,agent_name_="AgentX",model_provider_rag_="ollama",model_base_="http://localhost:11434",llm_model_name_="crewai-llama3:8b", embedding_model_="nomic-embed-text",**kwargs):
        self.llm_model_name_ = llm_model_name_
        self.model_provider_rag_ = model_provider_rag_
        self.embedding_model_ = embedding_model_
        self.model_provider_base_ = model_base_
        self.agent_name_= agent_name_
        for key, value in kwargs.items():
            setattr(self, key, value)
        ### create Agent RAG vector store
        self.embedding =OllamaEmbeddings(model=self.embedding_model_)
        self.vectorstore = Chroma(f"{ self.agent_name_}-memory",self.embedding  )

        #build components
        self.retriever = self.vectorstore.as_retriever()
        self.question_router = self.create_router()
        self.retrival_grader = self.create_retrival_grader()
        self.rag_chain = self.create_rag_prompt_chain()
        self.hallucination_grader = self.create_hallucination_grader()
        self.answer_grader = self.create_answer_grader()
        self.question_rewriter = self.create_re_writer()
        self.app = self.build_graph()


        os.environ["SERPER_API_KEY"] = "5fc96a6869c7f573b3c9972f8e291bc507b94494"
        self.web_search_tool = GoogleSerperAPIWrapper()
    
    def add_pdf(self,file_path):
        pdf_loader = PyMuPDFLoader(file_path)
        documents = pdf_loader.load()    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = text_splitter.split_documents(documents)
        self.vectorstore.add_documents(documents)
                       


    def add_to_vector_storage(self,documents):
        self.vectorstore.add_documents(documents)


    def add_web_content_to_rag(self,urls_):
        docs = [WebBaseLoader(url).load() for url in urls_]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(docs_list)
        self.add_to_vector_storage(doc_splits)

    def create_router(self):
        self.llm_router = ChatOllama(model=self.llm_model_name_, format="json", temperature=0)
        prompt = PromptTemplate(
            template="""You are an expert at routing a user question to a vectorstore or web search. \n
            Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. \n
            You do not need to be stringent with the keywords in the question related to these topics. \n
            Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
            Return the a JSON with a single key 'datasource' and no premable or explanation. \n
            Question to route: {question}""",
            input_variables=["question"],
        )
        return prompt | self.llm_router | JsonOutputParser()


    def create_retrival_grader(self):
        self.llm_retrival_grader= ChatOllama(model=self.llm_model_name_, format="json", temperature=0)

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n
            If the document contains keywords related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
            input_variables=["question", "document"],
        )
        return prompt | self.llm_retrival_grader | JsonOutputParser()

    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def create_rag_prompt_chain(self):
        self.rag_prompt = hub.pull("rlm/rag-prompt")
        self.llm_rag_prompt = ChatOllama(model=self.llm_model_name_, temperature=0)
        return  self.rag_prompt  | self.llm_rag_prompt  | StrOutputParser()

    def create_hallucination_grader(self):
        self.llm_hallucination = ChatOllama(model=self.llm_model_name_, format="json", temperature=0)
        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}
            Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            input_variables=["generation", "documents"],
        )
        return prompt | self.llm_hallucination | JsonOutputParser()
    
    def create_answer_grader(self):
        self.llm_answer = ChatOllama(model=self.llm_model_name_, format="json", temperature=0)
        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
            Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question}
            Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            input_variables=["generation", "question"],
        )

        return prompt | self.llm_answer  | JsonOutputParser()

    def create_re_writer(self):
        self.llm_rewriter = ChatOllama(model=self.llm_model_name_, temperature=0)
        re_write_prompt = PromptTemplate(
            template="""You a question re-writer that converts an input question to a better version that is optimized \n 
            for vectorstore retrieval. Look at the initial and formulate an improved question. \n
            Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
            input_variables=["generation", "question"],
        )
        return re_write_prompt | self.llm_rewriter  | StrOutputParser()


    def retrieve(self,state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}


    def generate(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(self,state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrival_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}


    def transform_query(self,state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}


    def web_search(self,state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}


### Edges ###


    def route_question(self,state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        print(question)
        source = self.question_router.invoke({"question": question})
        print(source)
        print(source["datasource"])
        if source["datasource"] == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source["datasource"] == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"


    def decide_to_generate(self,state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"


    def grade_generation_v_documents_and_question(self,state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"



    def build_graph(self):
        workflow = StateGraph(self.GraphState)

        # Define the nodes
        workflow.add_node("web_search", self.web_search)  # web search
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query

        # Build graph
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        return workflow.compile()



    def generate_answer(self, question):

        # Run
        inputs = {"question": f"{question}"}
        for output in app.stream(inputs):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint("\n---\n")

            # Final generation
        pprint(value["generation"])
        return value["generation"]
