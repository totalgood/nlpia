## Virtualbox

If you don't have Linux but want some, VirtualBox is a great way to get with the program.

### Preheat the Oven

#### [Download VirtualBox](https://www.virtualbox.org/wiki/Downloads) 

Virtual Box is the software for a dashboard that allows you to run a small linux computer within your laptop!  Find the right package for your OS and install it:

- [Ubuntu 15.03](http://download.virtualbox.org/virtualbox/5.0.14/virtualbox-5.0_5.0.14-105127~Ubuntu~wily_amd64.deb)
- [any Linux](https://www.virtualbox.org/wiki/Linux_Downloads)
- [OSX](http://download.virtualbox.org/virtualbox/5.0.14/VirtualBox-5.0.14-105127-OSX.dmg)
- [Win64](http://download.virtualbox.org/virtualbox/5.0.14/VirtualBox-5.0.14-105127-Win.exe)

#### [Download Vagrant](https://www.vagrantup.com/downloads.html)

Vagrant is a system that allows you to automate a lot of the process involved in configuring and controlling Virtual Box machines on your laptop. 

Go to the Vagrant [download page](https://www.vagrantup.com/downloads.html) to find the installation package for your OS. Download the package and double click it to open it with your package manager (software installer). On Ubuntu, the "Software Center" will launch and you click either the orange "Install" or "Upgrade" button.

### While You Wait

If your box downloads quickly, you can skip to the bottom and [Get Going](#get-going)

Otherwise you can [set up accounts](#set-up-accounts) with GitHub, etc while you wait.

#### Set Up Accounts

- [GitHub](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwiU15m349_KAhVLy2MKHVy7C3YQFggdMAA&url=https%3A%2F%2Fgithub.com%2Fjoin&usg=AFQjCNF6nezHQWX1hKwEFQVYRrUheS9_Ig)
  - Keep track of your projects
    - "Undo" when you make mistakes
    - `git checkout answer` when you have trouble with a project
  - Your forks of others' projects
  - Give back to the class by sharing your edits, material, links, code
  - Pull Requests to the Hack University projects
- Join us on [Slack](hackuniversity.slack.com)
  - We'll use Slack for polls and questions during class
  - You might also enjoy [PDXData.Slack.com](https://hackuniversity.slack.com/)
- [SignUp for a Twitter account](https://twitter.com/signup)
  - You'll need it for the twip project
  - [Tweet @hobsonlane](http://bit.ly/huml-help) so I can follow you
  - [Follow @hackoregon](https://twitter.com/hackoregon)
  - Include #huml in your tweets with questions or comments (**H**ack **U**niversity **M**machine **L**earning)
    - bit.ly shortcut: [http://bit.ly/huml-help](http://bit.ly/huml-help)
  - Follow your classmates tweets and the #huml hashtag 
- Google Drive or Gmail account
  - Put your name into [this spreadsheet](https://docs.google.com/spreadsheets/d/19HvN07XSNjlWF3TwLnyCUsCwXGBGwu15TemvVSIDwiI/edit?usp=sharing)

#### Text Editor

You can survive with an ssh and XWin connection to an editor on the Vagrant Box (virtual machine). But if you'd like a bit higher bandwidth and the "native" feel of your OS, install your favorite text editor on IDE on your laptop. Here are some of the features of the most popular python editors: 

- [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=linux&code=PCC)
  - integrated git
  - integrated execution of python scripts
  - not as useful for editing other languages
- [Sublime Text 2 Free Edition](http://www.sublimetext.com/2)
  - some basic execution of python scripts
  - powerful regular expressions
  - fast, clean
  - useful plugins like linters for almost all languages
  - easily customizable
  - difficult to install and maintain plugins
- [Atom](https://atom.io/)
  - fast
  - open source
  - backed by GitHub and a favorite of Google developers
  - customizable
  - new, bleeding edge

#### Version Control

And you probably want a decent "diff" tool to compare text files. I like [Meld](http://meldmerge.org/).

You probably also want `git` installed locally and have a way of running it from a shell with "readline" (remembers your commands so you don't have to retype them). [Git-Bash](http://www.git-scm.com/downloads) from GitHub does all this for you. On Linux, you probably already have git installed and you definitely have a decent shell ;)

#### Windows

If I have to work on Windows, I always install

- [Sublime Text 2 Free Edition](install-sublime.md)
- [CygWin](http://cygwin.com/install.html)
- [Anaconda](https://www.continuum.io/downloads)
- [Git-Bash](http://www.git-scm.com/downloads)

#### OSX

- [Sublime Text 2 Free Edition](install-sublime.md)
- [Anaconda](https://www.continuum.io/downloads)
- [Git-Bash](http://www.git-scm.com/downloads)

#### Linux

On Linux I can usually get away with using the standard package manager (apt-get on Ubuntu) and [`pip`](https://pip.pypa.io/en/stable/installing/)

`sudo -H pip install -r requirements.txt`?

### Get Going

Once you've installed VirtualBox and Vagrant you can boot up your first Vagrant box. Below I've modified [Vagrant's instructions](https://www.vagrantup.com/docs/getting-started/) to use Bill McGair's [customized version](https://atlas.hashicorp.com/boxes/search?utf8=%E2%9C%93&sort=&provider=&q=HackOregonDST](http://datasciencetoolbox.org).

You don't want to do this over a slow Cafe WiFi, or on a laptop that is running out of hard drive space.  The `vagrant up` command below has to first download the DataScienceToolbox *.box file, and it's huge (850 MB). If you have an Ethernet jack handy you might **plug in** before running the `vagrant up` command... 

```bash
$ mkdir hackoregon-dst
$ cd hackoregon-dst
$ vagrant init bmcgair/hackoregon-dst
$ vagrant plugin install vagrant-vbguest
$ vagrant up
```

Follow the rest of the instructions for the [Data Science Toolbox](http://datasciencetoolbox.org/) to set up ipython notebook to run on Bill's virtualbox you just downloaded and booted. You might also like having the dsftcl bundle installed on your box, if you like using Linux shell to process lots of data quickly.

### Pro Tip (optional)

If you'd like others to be able to query your database or run and edit your ipython notebooks on your server running on your laptop, you just need to share the IP address of your Vagrant box with them. And you'll need to ensure that your Vagrant box is configured to use NAT and set up the host machine (your laptop) forward port 8888, 80, and 8000 (or whatever ports you need) through Vagrant (VirtualBox).

## Finally

Now you can review the [syllabus](syllabus.md) and start working on some problems.