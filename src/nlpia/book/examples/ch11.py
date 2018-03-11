import re

lat = r'([-]?[0-9]?[0-9][.][0-9]{2,10})'
lon = r'([-]?1?[0-9]?[0-9][.][0-9]{2,10})'
sep = r'[,/ ]{1,3}'
re_gps = re.compile(lat + sep + lon)
re_gps.findall('http://...maps/@34.0551066,-118.2496763...')
# [(34.0551066, -118.2496763)]
re_gps.findall("https://www.openstreetmap.org/#map=10/5.9666/116.0566")
# [('5.9666', '116.0566')]
groups = re_gps.findall("Zig Zag Cafe is at 45.344, -121.9431 on my GPS.")
# [('45.3440', '-121.9431')]

# FIXME: remove unicode characters in regex or use regexes that can handle them
# deg,min,sec: 34°02'47.5"  # the degree unicode character will CRASH ipython!
deg_sym = r'[ ]?(°|d|deg|degree|degrees)[ ]?'
min_sym = r"[ ]?('|m|min|minute|minutes)[ ]?"
sec_sym = r'[ ]?("|s|sec|second|seconds)[ ]?'
dms = re.compile(r'([-]?[0-9]?[0-9]' + deg_sym +
                 r'[0-6]?[0-9]' + min_sym +
                 r'[0-6]?[0-9][.]?[0-9]{0,9}' + sec_sym +
                 r')[ ]?,[ ]?' +
                 r'([-]?1?[0-9]?[0-9]' + deg_sym +
                 r'[0-6]?[0-9]' + min_sym +
                 r'[0-6]?[0-9][.]?[0-9]{0,9}' + sec_sym +
                 r')')
dms.findall('34°02\'47.5"')
# []
print('34°02\'47.5"')
# 34°02'47.5"
dms.findall('34d02m47.5"')
# []
dms.findall('34d02m47.5s')
# []


def extract_latlon(s):
    matches = dms.findall(s)
    if len(matches):
        return float(matches[-1][0]), float(matches[-1][-1])
    else:
        return None, s


us = r'(([01]?\d)[-/]([0123]?\d)([-/]([012]\d)?\d\d)?)'
re.findall(us, 'Santa came on 12/25/2017 and a star appeared 12/12')
# [('12/25/2017', '12', '25', '/2017', '20'), ('12/12', '12', '12', '', '')]

eu = r'(([0123]?\d)[-/]([01]?\d)([-/]([012]\d)?\d\d)?)'
re.findall(eu, 'Alan Mathison Turing OBE FRS (23/6/1912-7/6/1954) was an English computer scientist.')
[('23/6/1912', '23', '6', '/1912', '19'),
 ('7/6/1954', '7', '6', '/1954', '19')]

# Deal with 2-digit an d4-digit and even 1-digit years from Year 0  to 3999 AD
# And lets name the parts of our year so we can easily coerce it into a datetime object
yr_19xx = (
    r'\b(?P<yr_19xx>' +
    '|'.join('{}'.format(i) for i in range(30, 100)) +
    r')\b'
    )
yr_20xx = (
    r'\b(?P<yr_20xx>' +
    '|'.join('{:02d}'.format(i) for i in range(10)) + '|' +
    '|'.join('{}'.format(i) for i in range(10, 30)) +
    r')\b'
    )
yr_cent = r'\b(?P<yr_cent>' + '|'.join('{}'.format(i) for i in range(1, 40)) + r')\b'
yr_ccxx = r'\b(?P<yr_ccxx>' + '|'.join('{:02d}'.format(i) for i in range(0, 100)) + r')\b'
yr = (
    r'\b(?P<yr>' +
    yr_19xx + '|' + yr_20xx + '|(?P<yr_xxxx>(' + yr_cent + ')(' + yr_ccxx + '))' +
    r')\b'
    )
re.findall(yr, "0, 2000, 01, '08, 99, 1984, 2030/1970 85 47 `66")

day = r'|'.join('{:02d}|{}'.format(i, i) for i in range(1, 32))

mon_words = 'January February March April May June July ' \
    'August September October November December'
# mon = '|'.join('{}|{}|{}'.format(m, m[:4], m[:3]) for m in months.split())
mon = '|'.join('{}|{}|{}|{}|{:02d}'.format(
    m, m[:4], m[:3], i + 1, i + 1) for i, m in enumerate(mon_words.split()))

eu = r'\b((' + day + r')\b[-,/ ]{0,2}\b(' + mon + r')\b[-,/ ]{0,2}\b(' + yr + r'))\b'

re.findall(eu, '31 Oct, 1970 25/12/2017')
# [('31 Oct, 1970', '31', 'Oct', '1970', '19', '70'),
#  ('25/12/2017', '25', '12', '2017', '20', '17')]

# [('0', '', '0'), ('2000', '20', '00'), ('01', '', '01'), ('99', '9', '9'), ('1984', '19', '84'), ('2030', '20', '30'), ('1970', '19', '70')]
# re.findall(yr'0, 2000, 01, 99, 1984, 2030/1970 ')

eu = r'(([0123]?\d)[-/ ]([01]?\d|' + mon + r')((\,[ ]|[-/ ])([012]\d)?\d\d)?)'
re.findall(eu, 'Barack Hussein Obama II (born August 4, 1961) is an American politician...')
# <1> this catches year zero ("0")for the astronomical calendar
# <2> this catches year integers 0 through 3999


def token_dict(token):
    return dict(
        ORTH=token.orth_,
        POS=token.pos_,
        TAG=token.tag_,
        DEP=token.dep_,
        LEMMA=token.lemma_)


def doc_dataframe(doc):
    return pd.DataFrame([token_dict(tok) for tok in parsed_sent])


doc_dataframe(en_model("In 1541 Desoto met the Pascagoula."))
#          ORTH       LEMMA    POS  TAG    DEP
# 0          In          in    ADP   IN   prep
# 1        1541        1541    NUM   CD   pobj
# 2      Desoto      desoto  PROPN  NNP  nsubj
# 3         met        meet   VERB  VBD   ROOT
# 4         the         the    DET   DT    det
# 5  Pascagoula  pascagoula  PROPN  NNP   dobj
# 6           .           .  PUNCT    .  punct

from spacy.matcher import Matcher
matcher = Matcher(en_model.vocab)
pattern = [{'TAG': 'NNP'}, {'LEMMA': 'meet'}, {'IS_ALPHA': True, 'OP': '*'}, {'TAG': 'NNP'}]
matcher.add('meeting', None, pattern)
matcher(en_model())
