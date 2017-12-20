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
[('12/25/2017', '12', '25', '/2017', '20'), ('12/12', '12', '12', '', '')]

eu = r'(([0123]?\d)[-/]([01]?\d)([-/]([012]\d)?\d\d)?)'
re.findall(eu, 'Alan Mathison Turing OBE FRS (23/6/1912-7/6/1954) was an English computer scientist.')
[('23/6/1912', '23', '6', '/1912', '19'),
 ('7/6/1954', '7', '6', '/1954', '19')]

months = 'January February March April May June July ' \
    'August September Octover November December'
mon = '|'.join('{}|{}|{}'.format(m, m[:4], m[:3]) for m in months.split())
digit_0n = '|'.join('{:02d}|{}'.format(i, i) for i in range(1, 9))
day = digit_0n + '|' + '|'.join('{}'.format(i, i) for i in range(10, 31))
yr_yy = digit_0n + '|00|' + '|'.join('{}'.format(i, i) for i in range(10, 99))  # <1>
yr_cc = r'\b(' + '|'.join('{}|{:02d}'.format(i, i) for i in range(0, 39)) + r')?\b'  # <2>

eu = r'(([0123]?\d)[-/ ]([01]?\d|' + mon + r')((\,[ ]|[-/ ])([012]\d)?\d\d)?)'
re.findall(eu, 'Barack Hussein Obama II (born August 4, 1961) is an American politician...')
# <1> this catches year zero ("0")for the astronomical calendar
# <2> this catches year integers 0 through 3999
