""" Example python snippets and listing in [Chapter 6](http://bit.ly/ghnlpia)"""
# import pandas as pd
import numpy as np
from nlpia.loaders import get_data
from sklearn.decomposition import PCA

wv = get_data('word2vec')

"""
>>> from nlpia.loaders import get_data
>>> wv = get_data('word2vec')

>>> naive_vector = wv['woman'] + wv['Europe'] + wv[physics'] +\
...     wv['scientist']

>>> naive_vector
array([ 0.87109375, -0.08511353,  0.7817383 ,  0.25634766, -0.10058594,
        ...
        0.20800781,  0.06420898,  0.09033203,  0.8261719 , -0.2545166 ],
      dtype=float32)
>>> wv.similar_by_vector(naive_vector)
[('scientist', 0.7510349750518799),
 ('physicist', 0.7328184843063354),
 ('physics', 0.7083248496055603),
 ('theoretical_physicist', 0.6038039922714233),
 ('astrophysicist', 0.6009320020675659),
 ('mathematician', 0.5989038944244385),
 ('particle_physicist', 0.5962826013565063),
 ('Physicist', 0.5940043926239014),
 ('biochemist', 0.5833224058151245),
 ('physicists', 0.577854573726654)]

>>> mean_vector = naive_vector / 4
>>> mean_vector
array([ 0.21777344, -0.02127838,  0.19543457,  0.06408691, -0.02514648,
       ...
        0.05200195,  0.01605225,  0.02258301,  0.20654297, -0.06362915],
      dtype=float32)
      >>> wv.similar_by_vector(mean_vector)
[('scientist', 0.7510349750518799),
 ('physicist', 0.7328184843063354),
 ('physics', 0.7083248496055603),
 ('theoretical_physicist', 0.6038039922714233),
 ('astrophysicist', 0.6009320020675659),
 ('mathematician', 0.5989038944244385),
 ('particle_physicist', 0.5962826013565063),
 ('Physicist', 0.5940043926239014),
 ('biochemist', 0.5833224058151245),
 ('physicists', 0.577854573726654)]
>>> # dividing by a constant (4) doesn't affect the "meaning" of the word vector
"""

naive_vector = wv['woman'] + wv['Europe'] + wv['physics'] + wv['scientist']
# naive_vector
# [('scientist', 0.7510349750518799),
#  ('physicist', 0.7328184843063354),
#  ('physics', 0.7083248496055603),
#  ('theoretical_physicist', 0.6038039922714233),
#  ('astrophysicist', 0.6009320020675659),
#  ('mathematician', 0.5989038944244385),
#  ('particle_physicist', 0.5962826013565063),
#  ('Physicist', 0.5940043926239014),
#  ('biochemist', 0.5833224058151245),
#  ('physicists', 0.577854573726654)]
mean_vector = naive_vector / 4
# dividing by a constant (4) doesn't affect the "meaning" of the word vector
# [('scientist', 0.7510349750518799),
#  ('physicist', 0.7328184843063354),
#  ('physics', 0.7083248496055603),
#  ('theoretical_physicist', 0.6038039922714233),
#  ('astrophysicist', 0.6009320020675659),
#  ('mathematician', 0.5989038944244385),
#  ('particle_physicist', 0.5962826013565063),
#  ('Physicist', 0.5940043926239014),
#  ('biochemist', 0.5833224058151245),
#  ('physicists', 0.577854573726654)]


"""
>>> from sklearn.decomposition import PCA

>>> pca = PCA(n_components=1)
>>> pca = pca.fit(np.array([wv['woman'], wv['Europe'], wv['physics'], wv['scientist']]))
>>> principal_vector = pca.components_[0]
>>> principal_vector
array([-0.0164827 , -0.02972822,  0.10266991,  0.03096823,  0.02329508,
       ...
       -0.02685342, -0.04291598, -0.03147813,  0.00288789,  0.00074687]],
      dtype=float32)
>>> wv.similar_by_vector(principal_vector)
[('scientist', 0.7614008188247681),
 ('physicist', 0.6602839231491089),
 ('physics', 0.6359047293663025),
 ('geoscientist', 0.5981974005699158),
 ('Physicist', 0.5895508527755737),
 ('astrophysicist', 0.5820537805557251),
 ('theoretical_physicist', 0.5803098678588867),
 ('researcher', 0.5745860934257507),
 ('science', 0.570257306098938),
 ('atmospheric_chemist', 0.5682488679885864)]


>>> normalized_vector = mean_vector - principal_vector
>>> wv.similar_by_vector(normalized_vector)
[('Europe', 0.6612628698348999),
 ('woman', 0.6248263120651245),
 ('European', 0.5074512958526611),
 ('man', 0.4976682662963867),
 ('Landslide_derails_train', 0.4912060499191284),
 ('Europeans', 0.48451364040374756),
 ('teenage_girl', 0.4837772250175476),
 ('Kaku_Yamanaka', 0.47976022958755493),
 ('United_States', 0.4793187081813812),
 ('IG_BNK_;)_COUNTRY', 0.47915613651275635)]
>>> # Kaku Yamanaka was the 6th oldest person (113 years, 116 days) when she died April 5, 2008
>>> # So the physics has been taken out of the vector since that's the principal component 
"""


pca = PCA(n_components=1)
pca = pca.fit(np.array([wv['woman'], wv['Europe'], wv['physics'], wv['scientist']]))
principal_vector = pca.components_[0]
normalized_vector = mean_vector - principal_vector
# normalized_vector
# array([ 0.21777344, -0.02127838,  0.19543457,  0.06408691, -0.02514648,
#        ...
#         0.05200195,  0.01605225,  0.02258301,  0.20654297, -0.06362915],
#       dtype=float32)
# wv.similar_by_vector(normalized_vector)
# [('Europe', 0.6612628698348999),
#  ('woman', 0.6248263120651245),
#  ('European', 0.5074512958526611),
#  ('man', 0.4976682662963867),
#  ('Landslide_derails_train', 0.4912060499191284),
#  ('Europeans', 0.48451364040374756),
#  ('teenage_girl', 0.4837772250175476),
#  ('Kaku_Yamanaka', 0.47976022958755493),
#  ('United_States', 0.4793187081813812),
#  ('IG_BNK_;)_COUNTRY', 0.47915613651275635)]
# Kaku Yamanaka was the 6th oldest person (113 years, 116 days) when she died April 5, 2008
# So the physics has been taken out of the vector since that's the principal component 

"""
>>> sentence = "Madam Curie was a physicist in Europe that discvered radiation ."
>>> vectors = [wv[w] for w in "Madam Curie was a physicist in Europe that discvered radiation .".split() if w in wv]
>>> np.sum(vectors)
0.16493225
>>> vectors = [wv[w] for w in "Madam Curie was a physicist in Europe that discvered radiation .".split() if w in wv]
>>> len(vectors)
8
>>> tokens = [t for t in sentence.split() if t in wv]
>>> tokens
['Madam', 'Curie', 'was', 'physicist', 'in', 'Europe', 'that', 'radiation']
>>> vectors = [wv[t] for t in tokens if t in wv]
>>> np.sum(vectors, axis=1)
array([ 2.2264786 , -3.1901474 ,  5.046369  ,  3.599121  , -0.50735474,
       -1.8962059 , -0.46976376, -4.643565  ], dtype=float32)
>>> np.sum(vectors, axis=0)
array([ 0.10473633,  0.6467285 ,  1.7527771 ,  0.4711914 , -0.17871094,
       ...
        0.18334961, -0.6008301 ,  0.17276001,  1.1404419 , -0.4885254 ],
      dtype=float32)
>>> naive_vector = np.sum(vectors, axis=0)
>>> len(vectors)
8
>>> mean_vector = naive_vector / len(vectors)
>>> wv.similar_by_vector(naive_vector)
[('physicist', 0.6525840759277344),
 ('radiation', 0.6193513870239258),
 ('Iodine_counteracts', 0.5717025399208069),
 ('physicist_Max_Planck', 0.5682262182235718),
 ('Henri_Becquerel', 0.5668175220489502),
 ('Curie', 0.5608154535293579),
 ('Climate_Experiment_SORCE', 0.5348742008209229),
 ('Wilhelm_Conrad_Roentgen', 0.5333640575408936),
 ('Gliese_###G', 0.5308289527893066),
 ('Pierre_Curie', 0.5277829170227051)]
>>> wv.similar_by_vector(mean_vector)
[('physicist', 0.6525840759277344),
 ('radiation', 0.6193513870239258),
 ('Iodine_counteracts', 0.5717025399208069),
 ('physicist_Max_Planck', 0.5682262182235718),
 ('Henri_Becquerel', 0.5668175220489502),
 ('Curie', 0.5608154535293579),
 ('Climate_Experiment_SORCE', 0.5348742008209229),
 ('Wilhelm_Conrad_Roentgen', 0.5333640575408936),
 ('Gliese_###G', 0.5308289527893066),
 ('Pierre_Curie', 0.5277829170227051)]

>>> normalized_vector = mean_vector - pca.components_
>>> mean_vector
array([ 0.01309204,  0.08084106,  0.21909714,  0.05889893, -0.02233887,
        ...
        0.0229187 , -0.07510376,  0.021595  ,  0.14255524, -0.06106567],
      dtype=float32)
>>> pc.components_ - np.mean(vectors, axis=0)
>>> pca.components_ - np.mean(vectors, axis=0)
array([[-6.95613474e-02, -7.88888931e-02, -1.08161360e-01,
        ...
        -1.90077946e-02, -8.51329267e-02,  7.61829540e-02]], dtype=float32)
>>> normalized_vector
array([[ 6.95613474e-02,  7.88888931e-02,  1.08161360e-01,
        ...
         1.90077946e-02,  8.51329267e-02, -7.61829540e-02]], dtype=float32)

>>> pca.components_
array([[-0.0564693 ,  0.00195217,  0.11093578,  0.00514601, -0.02890328,
        ...
        -0.02441795, -0.04301837,  0.00258721,  0.05742231,  0.01511728]],
      dtype=float32)
>>> wv.similar_by_vector(pca.components_[0])
[('radiation', 0.7282940745353699),
 ('physicist', 0.7270867228507996),
 ('Radiation', 0.6311361193656921),
 ('radiation_exposure', 0.5843555927276611),
 ('radioactivity', 0.5781536102294922),
 ('ionizing_radiation', 0.5744034051895142),
 ('Physicist', 0.5621669292449951),
 ('neutron_radiation', 0.551406741142273),
 ('particle_physicist', 0.5465821027755737),
 ('Kathryn_Higley', 0.5426877737045288)]
>>> wv.similar_by_vector(normalized_vector[0])
[('Madam', 0.685753583908081),
 ('FOREIGN_MINISTER_Thank', 0.5333032608032227),
 ('dear_madam', 0.5110944509506226),
 ('Sokenu', 0.504747211933136),
 ('was', 0.5007433891296387),
 ('Gifty', 0.48864680528640747),
 ('Anna_Bossman', 0.48664140701293945),
 ('Madame', 0.4832320511341095),
 ('Kwaku_Appiah', 0.4804612994194031),
 ('Enyonam', 0.4777677059173584)]
"""
