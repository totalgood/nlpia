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
