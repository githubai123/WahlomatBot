import MyAdaptiveRagAgent
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

########################################################################
Smarty = MyAdaptiveRagAgent.MyAdaptiveRagAgent()
urls = [
    "https://www.cdu.de/ueber-uns/geschichte-der-cdu",
]
Smarty.add_web_content_to_rag(urls)
Smarty.add_pdf("data/CDU.pdf")
Smarty.generate_answer("What is the CDU?")

'\n---\n'
# ---RETRIEVE---
# "Node 'retrieve':"
# '\n---\n'
# ---CHECK DOCUMENT RELEVANCE TO QUESTION---
# ---GRADE: DOCUMENT NOT RELEVANT---
# ---GRADE: DOCUMENT NOT RELEVANT---
# ---GRADE: DOCUMENT NOT RELEVANT---
# ---GRADE: DOCUMENT NOT RELEVANT---
# ---ASSESS GRADED DOCUMENTS---
# ---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---
# "Node 'grade_documents':"
# '\n---\n'
# ---TRANSFORM QUERY---
# better question: I see what you did there!

# Indeed, the initial question "What is the CDU?" can be re-written to a more specific and optimized query for vector-based retrieval systems. Here's my attempt:

# **Improved question:** `Definition of CDU`

# Rationale:

# 1. **Specificity**: By asking for a definition, we're narrowing down the scope of the answer, making it easier for the search algorithm to retrieve relevant results.
# 2. **Keyword emphasis**: Using "CDU" as a keyword in the query helps the search engine understand that this is the primary term of interest.
# 3. **Query structure**: The simple, concise structure of the improved question makes it more likely to be matched by relevant documents.

# This re-written question should lead to more accurate and relevant results when searching for information about CDU (which I assume stands for something like "Common Data Unit" or "Centralized Data Utility", but without more context, I couldn't pinpoint a specific meaning).. Improved question with no preamble:.. Improved question with no preamble:
# "Node 'transform_query':"
# '\n---\n'
# ---RETRIEVE---
# "Node 'retrieve':"
# '\n---\n'
# ---CHECK DOCUMENT RELEVANCE TO QUESTION---
# ---GRADE: DOCUMENT RELEVANT---
# ---GRADE: DOCUMENT RELEVANT---
# ---GRADE: DOCUMENT RELEVANT---
# ---GRADE: DOCUMENT RELEVANT---
# ---ASSESS GRADED DOCUMENTS---
# ---DECISION: GENERATE---
# "Node 'grade_documents':"
# '\n---\n'
# ---GENERATE---
# ---CHECK HALLUCINATIONS---
# ---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---
# ---GRADE GENERATION vs QUESTION---
# ---DECISION: GENERATION ADDRESSES QUESTION---
# "Node 'generate':"
# '\n---\n'
# ('Based on the retrieved context, I understand that the task is to answer '
#  "questions related to the CDU's (Christian Democratic Union) education "
#  'policy. The provided text appears to be a section from their party program, '
#  'outlining their goals and initiatives for education.\n'
#  '\n'
#  'To better assist you in answering your question, could you please provide '
#  "more context or specify what aspect of the CDU's education policy you would "
#  'like me to focus on?')


#####

score = Smarty.language_determinator.invoke(
                {"document": "Was ist dies für eine Sprache"})
# 
score = Smarty.language_determinator.invoke(
                {"document": "What is this language?"})

print(score)

score = Smarty.language_determinator.invoke(
                {"document": "Nous n'aimons pas le chocolat."})

print(score)

#######################################################
# {'language': 'German'}
# {'language': 'English'}
# {'language': 'French'}