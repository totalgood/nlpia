import spacy
import pandas as pd
# import numpy as np
# import annoy


########################################################################
# UCSD DHDS L8: NLP

nlp = spacy.load("en_core_web_lg")
docs = """
    Your blood pressure is too ___ .
    How old are ________ ?
    Does it hurt when you ________ ?
    Do you have a family history of ________ ?
    Dr. ________ told me the diagnosis .
    Patient LDL level is 100, ________ level is 50, and the total is ______ .            .
    Dr. Smith gave me ________ best estimate .
    """.splitlines()
docvecs = pd.DataFrame(
    (pd.DataFrame(
        [w.vector for w in nlp(doc.strip())]
        ).mean(axis=0)
     for doc in docs if len(doc.strip()))).round(3)

# UCSD DHDS L8: NLP
########################################################################
