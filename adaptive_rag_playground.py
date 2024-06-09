import AdaptiveRagAgent
from PoliticianAgent import PoliticianAgent
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import UnstructuredExcelLoader
import pandas as pd

##### Lose test script for individual components 
# Test new derived class 
#documents = ["data/CDU.pdf","data/FDP.pdf","data/Gruene.pdf"]
#bob = PoliticianAgent("bob", documents)
#
#exit(0)

##################      Get Data from Wahlomat          ################
########################################################################
#if(False):#
dataset_questions_file = "data/wahlomat_official_2024.xlsx"
df = pd.read_excel(dataset_questions_file, sheet_name="Datensatz EU 2024", usecols='B,F,G,H',skiprows=0)
unique_list = list(set(df["These: These"]))
filtered_df = df[df['Partei: Kurzbezeichnung'] == "CDU / CSU"]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(filtered_df)
   ###### Check Licence
print ("Fragen des WahlOmats:")
for id,Frage in enumerate(unique_list):
    print("{}: {}".format(id,Frage))




########################################################################
Smarty = AdaptiveRagAgent.AdaptiveRagAgent()



if (True):
    #urls = [
     #   "https://www.cdu.de/ueber-uns/geschichte-der-cdu",
    #]
   # Smarty.add_web_content_to_rag(urls)
    Smarty.add_pdf(["data/CDU.pdf"])
    question = "was ist die Position der CDU zur These : Die EU soll Atomkraft weiterhin als nachhaltige Energiequelle einstufen." #Smarty.translator.invoke({"text":"was ist die Position der CDU zur These : Die EU soll Atomkraft weiterhin als nachhaltige Energiequelle einstufen. ?","language":"English"})
    Smarty.generate_answer("question")

#'\n---\n'
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

text_sample =  "In welcher Sprache ist dieser Text geschrieben?"

score = Smarty.language_determinator.invoke({"document": text_sample})
# 
translation = Smarty.translator.invoke({"language":"French","text":text_sample})
print(translation)
translation = Smarty.translator.invoke({"language":"German","text":translation})
print(translation)
translation = Smarty.translator.invoke({"language":"English","text":translation})
print(translation)

####Language Determinator
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