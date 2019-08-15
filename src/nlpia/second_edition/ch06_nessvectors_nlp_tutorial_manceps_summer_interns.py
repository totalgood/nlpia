import spacy
import pandas as pd
import numpy as np
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

########################################################################
# NLP Tutorial Nessvectors

v_nurse = nlp('nurse').vector
v_doctor = nlp('doctor').vector

v_female = pd.DataFrame([w.vector for w in nlp('woman woman female female lady girl daughter mother sister she she her her')]).mean(axis=0)
v_male = pd.DataFrame([w.vector for w in nlp('man man male male gentleman boy son father brother he he him him')]).mean(axis=0)

female_doctor = v_doctor.dot(v_female) / np.linalg.norm(v_doctor) / np.linalg.norm(v_female)
female_nurse = v_nurse.dot(v_female) / np.linalg.norm(v_nurse) / np.linalg.norm(v_female)
male_doctor = v_doctor.dot(v_male) / np.linalg.norm(v_doctor) / np.linalg.norm(v_male)
male_nurse = v_nurse.dot(v_male) / np.linalg.norm(v_nurse) / np.linalg.norm(v_male)

v_nurse.dot(v_female) / np.linalg.norm(v_nurse) / pd.np.linalg.norm(v_female)
v_female_1 = nlp('female').vector
v_male_1 = nlp('male').vector
v_doctor.similarity(v_male)
tok_doctor = nlp('doctor')
tok_male = nlp('male')
tok_doctor.similarity(tok_male)
v_doctor.dot(v_male) / pd.np.linalg.norm(v_doctor) / pd.np.linalg.norm(v_male)
v_doctor.dot(v_male_1) / pd.np.linalg.norm(v_doctor) / pd.np.linalg.norm(v_male_1)

# NLP Tutorial Nessvectors
########################################################################
