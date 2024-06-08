import MyAdaptiveRagAgent
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

########################################################################
Smarty = MyAdaptiveRagAgent.MyAdaptiveRagAgent()
#urls = [
 #   "https://www.cdu.de/ueber-uns/geschichte-der-cdu",
#]
#Smarty.add_web_content_to_rag(urls)
#Smarty.add_pdf("data/CDU.pdf")
#Smarty.generate_answer("What is the CDU?")



#####

score = Smarty.language_determinator.invoke(
                {"document": "Was ist dies für eine Sprache"})

print(score)

score = Smarty.language_determinator.invoke(
                {"document": "What is this language?"})

print(score)

score = Smarty.language_determinator.invoke(
                {"document": "Nous n'aimons pas le chocolat."})

print(score)

#######################################################