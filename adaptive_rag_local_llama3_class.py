import MyAdaptiveRagAgent

########################################################################
Smarty = MyAdaptiveRagAgent.MyAdaptiveRagAgent()
urls = [
    "https://www.cdu.de/ueber-uns/geschichte-der-cdu",
]
Smarty.add_web_content_to_rag(urls)
Smarty.add_pdf("data/CDU.pdf")
Smarty.generate_answer("Was bedeutet CDU?")
#######################################################