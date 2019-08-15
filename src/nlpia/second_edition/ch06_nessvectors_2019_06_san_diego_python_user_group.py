""" San Diego Python User Group "Nessvector" Talk """

from nlpia.book.examples.ch06_nessvectors import *
wv = WV

#########################################################################################
# Trying to find Ada Lovelace the inventor of the first computer programming language Ada
# But instead turned up "Python_programming" and Telemundo (because W2V trained on recent Google News)
wv['famous'] + wv['woman'] + wv['computer'] + wv['programming'] + wv["60's"]
# KeyError: "word '60's' not in vocabulary"
wv['famous'] + wv['woman'] + wv['computer'] + wv['programming'] + wv["sixties"]
# array([ 4.21630859e-01, -8.24584961e-02,  3.02246094e-01,  8.33007812e-01, ...
wv.similar_by_vector(wv['famous'] + wv['woman'] + wv['computer'] + wv['programming'] + wv["sixties"])
# [('sixties', 0.5997753143310547),
#  ('forties_fifties', 0.5633885860443115),
#  ('seventies', 0.5617173910140991),
#  ('computer', 0.5398091673851013),
#  ('T_shirt_satirizing', 0.5329967737197876),
#  ('Jennifer_Ringley', 0.5260776281356812),
#  ('eighties', 0.5252149701118469),
#  ('owner_Janice_Maffucci', 0.5206712484359741),
#  ('Komando_hosts_national', 0.518700897693634),
#  ('Arrendondo_cook', 0.5176279544830322)]
wv.similar_by_vector(wv['woman'] + wv['computer'] + wv['programming'] + +wv['language'] + wv["1960"])
# KeyError: "word '1960' not in vocabulary"
wv.similar_by_vector(wv['woman'] + wv['computer'] + wv['programming'] + +wv['language'])
# [('language', 0.68879234790802),
#  ('programming', 0.6577489972114563),
#  ('computer', 0.617863118648529),
#  ('programing', 0.587421178817749),
#  ('Python_programming', 0.5564959645271301),
#  ('langauge', 0.5470107793807983),
#  ('broadcaster_Telemundo', 0.5313897132873535),
#  ('Komando_hosts_national', 0.5189266204833984),
#  ('novelas_movies', 0.5128160715103149),
#  ('Optional_subtitles', 0.511816680431366)]
wv.similar_by_vector(wv['woman'] + wv['computer'] + wv['programming'] + wv['ada'] + wv['CS'])
wv.similar_by_vector(wv['woman'] + wv['computer'] + wv['programming'] + wv['first'] +
                     wv['language'] + wv['ADA'] + wv['cs'] + wv['woman'] + wv['supercomputer'])
# [('computer', 0.5967065095901489),
#  ('supercomputer', 0.5708476305007935),
#  ('includes_searchable_printable', 0.5467446446418762),
#  ('Python_programming', 0.546095073223114),
#  ('Arrendondo_cook', 0.5306745171546936),
#  ('Seeking_LPN', 0.525610625743866),
#  ('dell_laptop', 0.5171411037445068),
#  ('Komando_hosts_national', 0.5147794485092163),
#  ('woman', 0.5145055651664734),
#  ('searchable_printable', 0.5136831402778625)]

""" More Ada Lovelace
>>> wv.similar_by_vector(wv['woman'] + wv['computer'] + wv['programming'] + wv['ADA'] + wv['cs'] + wv['woman'])
[('woman', 0.6299353241920471),
 ("QI'ma", 0.5150842666625977),
 ('dell_laptop', 0.512478232383728),
 ('computer', 0.5108146667480469),
 ('magic_quotes_gpc', 0.5055264234542847),
 ('Alleged_molester', 0.5006985664367676),
 ('girl', 0.5000299215316772),
 ('Visit_www.xmradio.ca', 0.4956779479980469),
 ('Herre_Bagwell', 0.49191924929618835),
 ('includes_searchable_printable', 0.4893316328525543)]
>>> wv.similar_by_vector(wv['woman'] + wv['computer'] + wv['programming'] + wv['first'] + wv['language'] + wv['ADA'] + wv['cs'] + wv['woman'])
[('woman', 0.5785168409347534),
 ('language', 0.5715309381484985),
 ('includes_searchable_printable', 0.5487347841262817),
 ('Seeking_LPN', 0.5263922214508057),
 ('searchable_printable', 0.5251418948173523),
 ('Jamaican_Patois', 0.5142751932144165),
 ("QI'ma", 0.5138590335845947),
 ('Optional_subtitles', 0.5136313438415527),
 ('Autism_impairs', 0.511799156665802),
 ('Visit_www.xmradio.ca', 0.5117049217224121)]
>>> wv.similar_by_vector(wv['woman'] + wv['computer'] + wv['programming'] + wv['ada'] + wv['CS'])
[('computer', 0.5700141787528992),
 ('logo_inserter', 0.5551257133483887),
 ('Visit_www.xmradio.ca', 0.5547385215759277),
 ('CNA_ir', 0.5507899522781372),
 ('ada', 0.5451924204826355),
 ('ActiveX_COM', 0.5448037981987),
 ('dell_laptop', 0.5418514013290405),
 ('google_chrome', 0.5369359850883484),
 ('Komando_hosts_national', 0.535258412361145),
 ('ensnare_accelerators', 0.5325168967247009)]"""

wv.similar_by_vector(wv['eighties'] + wv['reality'] + wv['bites'] + wv['california'] + wv['actress'])
# [('california', 0.7123905420303345),
#  ('sandra_bullock', 0.6160793304443359),
#  ('brad_pitt', 0.6131527423858643),
#  ('kristen_stewart', 0.6061184406280518),
#  ('lindsay_lohan', 0.5995123386383057),
#  ('hollywood', 0.5960849523544312),
#  ('taylor_swift', 0.5952141880989075),
#  ('miley_cyrus', 0.5918698310852051),
#  ('megan_fox', 0.5807573199272156),
#  ('kate_moss', 0.5733609199523926)]
wv.similar_by_vector(wv['eighties'] + wv['Reality_Bites'] + wv['stranger'] + wv['things'] + wv['Alien'] + wv['actress'] + wv['woman'])
# [('kristen_stewart', 0.6247533559799194),
#  ('Bjork_swan_dress', 0.6234749555587769),
#  ('sandra_bullock', 0.6186246871948242),
#  ('Reality_Bites', 0.6159488558769226),
#  ('zac_efron', 0.612686812877655),
#  ('taylor_swift', 0.6087606549263),
#  ('Kate_Beckinsdale', 0.6080144643783569),
#  ('tells_Parade.com', 0.605420708656311),
#  ('Dear_Incredible_Inman', 0.6049702763557434),
#  ('Baywatch_Nights', 0.6029472947120667)]
wv.similar_by_vector(wv['eighties'] + wv['Reality_Bites'] + wv['stranger'] + wv['things'] + wv['Alien'] + wv['actress'] + wv['woman'] + wv['Winona'])
# [('Bjork_swan_dress', 0.6245858669281006),
# ('AnnaSophia', 0.6119588613510132),
# ('Reality_Bites', 0.6097363233566284),
# ('actress', 0.6083126068115234),
# ('Femme_fatale', 0.6049113869667053),
# ('Kirsten_Dunst_Keira_Knightley', 0.5990383625030518),
# ('Ginnifer', 0.5986188650131226),
# ('Judy_Carne', 0.598508358001709),
# ('Susan_Anspach', 0.5971428155899048),
# ('Epatha', 0.596214771270752)]
wv.similar_by_vector(wv['eighties'] + wv['Reality_Bites'] + wv['stranger'] + wv['things'] +
                     wv['Alien'] + wv['actress'] + wv['woman'] + wv['Winona_Ryder'])
# [('Winona_Ryder', 0.6972660422325134),
#  ('actress', 0.6614755988121033),
#  ('Hysterical_Blindness_HBO', 0.6591032147407532),
#  ('Freddie_Prinze_Jr', 0.6499183177947998),
#  ('Bjork_swan_dress', 0.642225444316864),
#  ('Susan_Anspach', 0.6415334939956665),
#  ('Femme_fatale', 0.6389787793159485),
#  ('Kate_Beckinsdale', 0.6389024257659912),
#  ('Judy_Carne', 0.638439416885376),
#  ('Kristen_Dunst', 0.6346170902252197)]
wv.similar_by_vector(wv['eighties'] + wv['Reality_Bites'] + wv['stranger'] + wv['things'] +
                     wv['Alien'] + wv['actress'] + wv['woman'] + wv['Femme_fatale'])
# [('Femme_fatale', 0.6835325956344604),
#  ('Bjork_swan_dress', 0.655756950378418),
#  ('Freddie_Prinze_Jr', 0.6435261368751526),
#  ('actress', 0.6423774361610413),
#  ('Hysterical_Blindness_HBO', 0.6377550959587097),
#  ('AnnaSophia', 0.6363706588745117),
#  ('Crave_Online_Are', 0.6325239539146423),
#  ('Judy_Carne', 0.6322861909866333),
#  ('Kate_Beckinsdale', 0.6313980221748352),
#  ('Kirsten_Dunst_Keira_Knightley', 0.6287360787391663)]

wv.similar_by_vector(wv['Reality_Bites'] + wv['stranger'] + wv['things'] + wv['Alien'] + wv['actress'] + wv['woman'] + wv['Winona_Ryder'])
wv.similar_by_vector(wv['Reality_Bites'] + wv['stranger'] + wv['things'] + wv['Alien'] + wv['actress'] + wv['woman'])
wv.similar_by_vector(wv['Reality_Bites'] + wv['stranger'] + wv['things'] + wv['Alien_Resurrection'] + wv['actress'] + wv['woman'])
wv.similar_by_vector(wv['Reality_Bites'] + wv['stranger'] + wv['things'] + wv['Alien_Resurrection'] + wv['actress'] + wv['woman'], wv['Minnesota'])
wv.similar_by_vector(wv['Reality_Bites'] + wv['stranger'] + wv['things'] + wv['Alien_Resurrection'] + wv['actress'] + wv['woman'] + wv['Minnesota'])
wv.similar_by_vector(wv['Reality_Bites'] + wv['Alien_Resurrection'] + wv['actress'] + wv['woman'] + wv['Minnesota'])
wv.similar_by_vector(wv['Reality_Bites'] + wv['Alien_Resurrection'] + wv['cast'])
wv.similar_by_vector(wv['Reality_Bites'] + wv['cast'] + wv['actress'])
wv.similar_by_vector(wv['Reality_Bites'] + wv['cast'] + wv['actress'] + wv['Rat_Pack'])
wv.similar_by_vector(wv['Star Wars'] + wv['cast'] + wv['actress'] + wv['Rat_Pack'])
wv.similar_by_vector(wv['Star_Wars'] + wv['cast'] + wv['actress'] + wv['Rat_Pack'])
wv.similar_by_vector(wv['Star_Wars'] + wv['cast'] + wv['actress'])
wv.similar_by_vector(wv['Star_Wars'] + wv['cast'] + wv['actress'] - wv['movie_title'])
wv.similar_by_vector(wv['Star_Wars'] + wv['cast'] + wv['actress'] - wv['movie'])
wv.similar_by_vector(wv['Star_Wars'] + wv['cast'] + wv['actress'] - wv['movie'] - wv['cinema'] + wv['name'] + wv['person'])
wv.similar_by_vector(wv['Star_Wars'] + wv['cast'] + wv['castmember'] + wv['actress'] - wv['movie'] - wv['cinema'] + wv['name'] + wv['person'])
wv.similar_by_vector(3 * wv['Star_Wars'] + wv['cast'] + wv['castmember'] + wv['actress'] - wv['movie'] - wv['cinema'] + wv['name'] + wv['person'])
wv.similar_by_vector(wv['Star_Wars'] + wv['cast'] + wv['castmember'] + wv['actress'] + wv['name'] + wv['person'])
wv.similar_by_vector(wv['Star_Wars'] + wv['Princess_Leia'] + wv['cast'] + wv['castmember'] + wv['actress'] + wv['name'] + wv['person'])

##########################################################################################
# OK let's try looking for an herbal remedy
wv.similar_by_vector(wv['native_american'] + wv['medicinal'] + wv['herb'])
wv.similar_by_vector(wv['native_American'] + wv['medicinal'] + wv['herb'])
wv.similar_by_vector(wv['Native_American'] + wv['medicinal'] + wv['herb'])
wv.similar_by_vector(wv['Native_American'] + wv['medicinal'] + wv['herb'] + wv['wound'] + wv['dressing'])
wv.similar_by_vector(wv['Native_American'] + wv['medicinal'] + wv['herb'] + wv['wound'] + wv['dressing'] + wv['common'] + wv['name'])
wv.similar_by_vector(wv['Native_American'] + wv['medicinal'] + wv['herb'] + wv['wound'] +
                     wv['dressing'] + wv['common'] + wv['name'] + wv['creek'] + wv['bank'] + wv['stem'])
wv.similar_by_vector(wv['Native_American'] + wv['medicinal'] + wv['herb'] + wv['wound'] + wv['dressing'] +
                     wv['common'] + wv['name'] + wv['creek'] + wv['bank'] + wv['stem'] + wv['friction'] + wv['drill'])
wv.similar_by_vector(wv['Native_American'] + wv['medicinal'] + wv['herb'] + wv['wound'] + wv['dressing'] + wv['common'] +
                     wv['name'] + wv['creek'] + wv['bank'] + wv['stem'] + wv['leaves'] + wv['leaf'] + wv['friction'] + wv['drill'])
tokens = 'Native_American medicinal herb wound dressing paralyze fish common name creek river bank stem leaves leaf friction drill'.split()
sum([wv[tok] for tok in tokens])
df = pd.DataFrame([wv[tok] for tok in tokens], index=tokens)
df
#                       0         1         2         3         4    ...       295       296       297       298       299
# Native_American -0.137695  0.025635  0.165039  0.161133 -0.118164  ...  0.396484  0.237305  0.030273  0.486328 -0.089355
# medicinal        0.102051  0.071777  0.087402  0.111816 -0.151367  ...  0.455078  0.122070  0.205078  0.378906  0.147461
# herb             0.072754  0.083008  0.061035  0.211914  0.246094  ...  0.154297  0.019897  0.361328  0.349609  0.271484
# wound            0.096191  0.292969  0.359375 -0.077637 -0.150391  ...  0.318359  0.093750 -0.091797  0.192383  0.136719
# dressing        -0.095703  0.257812 -0.082031 -0.024536  0.116211  ...  0.069336  0.357422 -0.040039  0.009766  0.345703
# paralyze         0.102051  0.043945  0.166016  0.287109 -0.205078  ...  0.067871 -0.431641 -0.259766 -0.049072  0.238281
# fish            -0.174805  0.159180  0.004150  0.075195 -0.097656  ...  0.014160 -0.065918 -0.018433  0.300781  0.085449
# common          -0.002686  0.013733  0.184570  0.328125 -0.107422  ...  0.161133  0.106445 -0.113281  0.275391  0.244141
# name             0.148438  0.152344  0.069336  0.047852 -0.137695  ... -0.092285  0.308594 -0.155273 -0.294922 -0.152344
# creek            0.042480 -0.001266 -0.095215 -0.062256  0.126953  ...  0.033203 -0.107422 -0.296875  0.322266 -0.052979
# river            0.008362  0.181641  0.089844  0.072266  0.118652  ...  0.034912 -0.037354 -0.255859  0.394531 -0.058594
# bank             0.021973  0.134766 -0.057861  0.055664  0.099121  ... -0.019287 -0.145508 -0.055664 -0.038330 -0.318359
# stem            -0.123535  0.119629 -0.048828 -0.291016 -0.376953  ...  0.042725 -0.105957 -0.077637  0.159180  0.067383
# leaves           0.257812  0.210938  0.096191  0.028198 -0.037842  ...  0.237305 -0.170898 -0.068848 -0.106934  0.182617
# leaf             0.084961  0.289062  0.030884  0.125000 -0.062500  ...  0.066895 -0.209961  0.194336 -0.036133  0.195312
# friction         0.220703 -0.017456 -0.112793 -0.190430 -0.398438  ...  0.160156  0.102051  0.031738  0.170898  0.071777
# drill           -0.131836 -0.066406  0.029175  0.118164  0.000957  ... -0.046387 -0.019531  0.053223 -0.149414 -0.187500

wv.similar_by_vector(df.iloc[:5].sum().values)
# [('medicinal', 0.7557034492492676),
#  ('herb', 0.7474513053894043),
#  ('herbal', 0.6829161643981934),
#  ('herbs', 0.6770135760307312),
#  ('decoctions', 0.6254310607910156),
#  ('medicinal_herbs', 0.6206021904945374),
#  ('medicinal_herb', 0.6140773296356201),
#  ('medicinally', 0.6109899878501892),
#  ('aromatic_herbs', 0.6025412082672119),
#  ('herbal_medicine', 0.5933672785758972)]
df.iloc[:5, :].sum().values
tokens = 'Native_American medicinal herb wound dressing common name creek bank stem leaves leaf friction drill'.split()
df = pd.DataFrame([wv[tok] for tok in tokens], index=tokens)
df
wv.similar_by_vector(df.iloc[:5].sum().values)
wv.similar_by_vector(df.iloc[:10].sum().values)
wv.similar_by_vector(df.sum().values)
wv.similar_by_vector(wv['Reality_Bites'] + wv['cast'] + wv['actress'])
wv.similar_by_vector(wv['Donnie_Darko'] + wv['cast'] + wv['actor'])
wv.similar_by_vector(wv['Pulp_Fiction'] + wv['cast'] + wv['actor'] + wv['woman'])
wv.similar_by_vector(wv['Pulp_Fiction'] + wv['cast'] + wv['actor'] + wv['woman'])
wv.similar_by_vector(wv['Pulp_Fiction'] + wv['cast'] + wv['actress'] + wv['woman'])
wv.similar_by_vector(wv['Blade_Runner'] + wv['name'] + wv['from'] + wv['cast'] + wv['actress'] + wv['woman'])
wv.similar_by_vector(wv['Native_American'] + wv['medicinal'] + wv['herb'] + wv['wound'] + wv['dressing'] + wv['common'] +
                     wv['name'] + wv['creek'] + wv['bank'] + wv['stem'] + wv['leaves'] + wv['leaf'] + wv['friction'] + wv['drill'])
# [('Discard_bay', 0.6322928071022034),
#  ('herb', 0.6217018961906433),
#  ('mullein', 0.6141064763069153),
#  ('sweet_woodruff', 0.6073442101478577),
#  ('milky_sap', 0.5956025123596191),
#  ('aromatic_herb', 0.5926221609115601),
#  ('sweet_marjoram', 0.5889462232589722),
#  ('Prickly_pear', 0.5827879309654236),
#  ('Mince_garlic', 0.5769100189208984),
#  ('tuberous_root', 0.5762436985969543)]
#
# "mullein" That's IT !!!!


###########################################################
# Now let's play with some `nessvectors`
nessvector('cat')
# placeness     -1.560885
# peopleness    -0.216057
# animalness     2.906886
# conceptness   -1.442183
# femaleness     0.312240
nessvector('herb')
# placeness     -0.340492
# peopleness     0.036387
# animalness     0.647465
# conceptness   -0.356363
# femaleness     0.013003
nessvector('dog')
# placeness     -1.453316
# peopleness    -0.589015
# animalness     3.112393
# conceptness   -1.225242
# femaleness     0.155180
nessvector('tiger')
# placeness     -1.283883
# peopleness    -0.140355
# animalness     2.350045
# conceptness   -1.342111
# femaleness     0.416304
