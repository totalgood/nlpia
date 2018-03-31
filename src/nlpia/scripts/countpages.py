#!/usr/bin/env python3
""" Renders *.asc (asciidoc) files to PDF or HTML then counts pages & words

HTML or other render formats requires asciidoctor` ruby package/cli:

```
$ brew install asciidoctor || gem install asciidoctor
```

PDF rendering requires the ruby package and cli command `asciidoctor-pdf`:

```bash
$ gem install asciidoctor-pdf --pre
```

You probably also want syntax highlighting so:

```
$ gem install rouge
$ gem install pygments.rb
$ gem install coderay
````

You can activate the Rouge syntax highlighter with the following atttribute in your *.asc files:

```asciidoc
:source-highlighter: rouge
```

Usage:

Put all images in a folder called `images` and at the same level a directory called `manuscript` with the *.asc files.
A `build` directory must also be present at the same level. Then to render ascidic files to PDF:

$ python countpages.py ~/src/lane/manuscript/ pdf


create a symlink (or hard link) in .git/hooks
>>> cd .git/hooks/
# >>> ln -s ../../code/utils/pre-commit
"""

import os
import subprocess
import sys

from traceback import format_exc


nlpia_dir = os.path.dirname(os.path.realpath(__file__))


def shell_quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


def parse_args(args=None):
    args = sys.argv[1:] if args is None else args
    manuscript_dir = os.path.abspath(os.path.expanduser(args[0] if len(args) > 0 else '.'))
    renderas = (args[1] if len(args) > 1 else 'html5').lower().strip()
    renderext = 'html' if renderas == 'html5' else renderas
    return {'manuscript_dir': manuscript_dir, 'renderas': renderas, 'renderext': renderext}


def render(manuscript_dir='manuscript', renderas='html5', renderext='html'):
    files = os.listdir(manuscript_dir)
    project_dir = os.path.dirname(manuscript_dir)
    build_dir = os.path.join(project_dir, 'build')
    css_path = os.path.join(build_dir, 'manning.css')
    asciidoctor = 'asciidoctor-pdf' if renderas == 'pdf' else 'asciidoctor'

    total_wc, chapters_wc = 0, 0
    for filename in files:
        if filename.lower().endswith('.asc'):
            print("-" * 40)
            print(filename + ':')
            quoted_filename = shell_quote(os.path.join(manuscript_dir, filename))
            result = subprocess.Popen("wc {}".format(quoted_filename),
                                      shell=True, stdout=subprocess.PIPE)
            try:
                wc = int(result.communicate()[0].split()[1])
                total_wc += wc
                if filename.lower().strip().startswith("'ch"):
                    chapters_wc += wc
                print('    Word count: {}'.format(wc))
                print('    Approximate Pages: {}'.format(int(wc / 400.) + 1))
            except:
                print('    ERROR: `wc` SHELL COMMAND FAILED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(format_exc())
            html_file_path = os.path.join(build_dir, filename[:-4] + '.' + renderext)
            shellcmd = "{} -a stylesheet={} -b {} -d book -B {} -o {} {}".format(
                asciidoctor, css_path, renderas, shell_quote(manuscript_dir),
                shell_quote(html_file_path), shell_quote(filename))
            print(shellcmd)
            result = subprocess.Popen(shellcmd,
                                      shell=True, stdout=subprocess.PIPE)
            result.wait()  # don't let commit proceed until rendering is done
            try:
                ans = ' '.join(result.communicate()[0])
                print('    asciidoctor returned: {}'.format(ans))
            except:
                print('    ERROR: `asciidoctor` SHELL COMMAND FAILED !!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('    You probably need to `gem install asciidoctor asciidoctor-pdf-cjk`')

    print('=' * 60)
    print('Total words, pages: {}, {}'.format(total_wc, int(total_wc / 400.) + 1))
    print('Total chapter words, pages: {}, {}'.format(chapters_wc, int(chapters_wc / 400.) + 1))


def commit(project_dir='..', renderext='HTML'):
    os.chdir(project_dir)
    shellcmd = "git add -f {}/build/*".format(project_dir)
    print(shellcmd)
    result = subprocess.Popen(shellcmd, shell=True, stdout=subprocess.PIPE)
    msg = result.communicate()[0]
    print('Committed newly-rendered asciidoc->{} files:'.format(renderext))
    print(msg)


def main():
    args = parse_args(sys.argv[1:])
    render(**args)
    commit(renderext=args['renderas'])


if __name__ == '__main__':
    main()
