f = zipfile.ZipFile('aiml-en-us-foundation-alice.v1-9.zip')
with f.open() as fin:
    for line in fin:
        print(line)

# want to strip of trailing </aiml> tag to be able to concatenate them all.
for name in f.namelist():
    with f.open(name) as fin:
        # print(name)
        happyending = '#!*@!!BAD'
        for i, line in enumerate(fin):
            try:
                line = line.decode('utf-8').strip()
            except UnicodeDecodeError:
                line = line.decode('ISO-8859-1').strip()
            if line.startswith('</aiml>'):
                happyending = (i, line)
                break
        if happyending != (i, line):
            print('!!!!BAD: ' + name)
            print(i, line)