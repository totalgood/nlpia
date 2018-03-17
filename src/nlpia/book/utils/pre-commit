#!/usr/bin/env python3

# To use: create a symlink (or hard link) in .git/hooks
# >>> cd .git/hooks/
# >>> ln -s ../../code/utils/pre-commit

import os
import subprocess

from traceback import format_exc

filename = os.path.dirname(os.path.realpath(__file__))

os.chdir('manuscript')
files = os.listdir()


def shell_quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


total_wc, chapters_wc = 0, 0
for filename in files:
    if filename.lower().endswith('.asc'):
        print("-" * 40)
        print(filename + ':')
        filename = shell_quote(filename)
        result = subprocess.Popen("wc {}".format(filename),
                                  shell=True, stdout=subprocess.PIPE)
        try:
            wc = int(result.communicate()[0].split()[1])
            total_wc += wc
            if filename.lower().strip().startswith("'ch"):
                chapters_wc += wc
            print('  Word count: {}'.format(wc))
            print('  Approximate Pages: {}'.format(int(wc / 400.) + 1))
        except:
            print('  WC SHELL COMMAND FAILED!!!!!')
            print(format_exc())
        result = subprocess.Popen("asciidoctor -a stylesheet=manuscript/manning.css -b html5 -d book -B .. -o build/{} {}".format(
                                  filename[:-4] + '.html', filename),
                                  shell=True, stdout=subprocess.PIPE)
        result.wait()  # don't let commit proceed until rendering is done
        try:
            ans = ' '.join(result.communicate()[0])
            print('asciidoctor returned: {}'.format(ans))
        except:
            print('  `asciidoctor` SHELL COMMAND FAILED!!!!!')
            print('  You probably need to `gem install asciidoctor`')

print('=' * 60)
print('Total words, pages: {}, {}'.format(total_wc, int(total_wc / 400.) + 1))
print('Total chapter words, pages: {}, {}'.format(chapters_wc, int(chapters_wc / 400.) + 1))


result = subprocess.Popen("cd .. && git add manuscript/build/*.html ; cd .git", shell=True, stdout=subprocess.PIPE)
msg = result.communicate()[0]
# print('Committed newly-rendered HTML by aciidoctor:')
# print(msg)
