# Contributing

Share your ideas with your fellow [NLP enthusiasts](AUTHORS)! This community-driven project is built on the contributions of NLPIA readers. 
We love

 - Bug reports (github issues)
 - Even better, bug fixes! (pull requests)
 - Documentation
 - Feature ideas
 - Even better, feature implementations! (pull requests)

You can get ideas from other readers [in the GitHub repo](https://github.com/totalgood/nlpia/issues). Feel free to jump on any of them by submitting a [pull request (PR)](https://github.com/totalgood/nlpia/pulls).
 
Our MIT [`LICENSE`](https://github.com/totalgood/nlpia/blob/master/LICENSE) doesn't **require** you to contribute your hard work, but you'll be a lot happier if you know that others are benefiting from your code.

## Bug Reports

 1. Please include a Python snippet or instructions for reproducing the problem.
 2. Attach or link to the data you used (if needed to recreate the bug)
 3. Explain the behavior you expected, and how what you got was different

## Pull Requests

 1. When you start working on a feature, first create new branch from the latest master commit [GitHub master](https://github.com/slackha/pyJac/tree/master): `git checkout origin master -b feature/my-feature-123`
 2. Your first commit should include docstrings and doctests in [Google/NumPy style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy)
 3. Use [pythonic style](http://www.python.org/dev/peps/pep-0008/)
 4. Superstars and professionals reference relevant GitHub issues in commit messages: `git commit -am "fixes #123"`

## Tests and Docs

Your aren't done until `python setup.py test` runs your code and passes. 
If you've written doctests in step 2 above, this this is as easy as pushing your commit and waiting for [Travis](travis-ci.org) to update the "tests pass" badge to green on your branch. 

Your reward for reading this is knowing that early contributors will get their "name in lights" in [the book](https://bit.ly/nlpiabook). 
You just need to get your PR in before it goes to print this Spring (2018)... and you get 42% off if you use the **discount code "sllane"** during checkout. 

And if you contribute to the parts of `nlpia` that are actually generating some of the text in the book... well, then you can say that you taught one of the authors how to write ;)

## Style Guide

Readable code has fewer bugs, but only if it's readable by others, even new python developers. 

Follow the patterns preferred by other python developers and projects that "play nice with others" and build great stuff. 
Quality open source projects are a great place to look for patterns for whatever problem you're trying to solve: 

* [Django](https://github.com/django/django)
* [numpy](https://github.com/numpy/numpy)
* [spaCy](https://github.com/explosion/spaCy)

And `import this` if you haven't already.

### *Explicit* is better than implicit

Avoid clever redirection, abstraction, and hidden "magic", if a more direct explicit approach will accomplish the same thing with few lines of code.

Override builtin methods like `__init__` and `__call__` and `__getitem__` and only for classes that inherit a class or ABC that already implements those methods.  This ensures the API for your class is intuitive to someone not looking at the class definition.
  

### [PEP8](https://www.python.org/dev/peps/pep-0008/) with these exceptions

Follow the [Hettinger interpretation of PEP8](https://www.youtube.com/watch?v=wf-BqAjZb8M) for beautiful, readable code.

* max line length is "about" 120 chars (if you go a little over, don't worry about it). Modern displays can handle it. 
* max complexity: "mccabe_threshold": 12,  # McCabe complexity checker limits the number of conditional branches
* use double quotes for natural language strings: `"Hello {name}!}".format(name="Chloe")`
* use single quotes machine-readable strings: `commands['describe']['default'] = "I can't see anything."`
* type hints are a good idea on conditional branches that are rarely run, but duck-typing is preferred for mainline code

### My Sublime Settings

I use the [Sublime Anaconda IDE](https://damnwidget.github.io/anaconda/) to lint and auto-fixer my python. 

The Sublime Anaconda IDE is not the same as the company that maintains the awesome `conda` python package manager. 
Sublime Anaconda is a sublime package you can only install within Sublime Text 3, like [this](http://damnwidget.github.io/anaconda/#anaconda-overview-out-of-the-box). 

Below are my Sublime Anaconda IDE settings:

#### `$HOME/Library/Application Support/Sublime Text 3/Packages/User/Anaconda.sublime-settings`

```javascript
{
    // Maximum McCabe complexity (number of conditional branches within a function).
    "mccabe_threshold": 7,

    // Maximum line length
    "pep8_max_line_length": 120

    "anaconda_linter_show_errors_on_save": false,
    "use_pylint": false,
    "pep8_rcfile": false,
    
    // Linting PEP8 rules to ignore. Anaconda autolint can't safely fix these on save).
    "pep8_ignore":
    [
        "E309", "E123"
    ],
}
```




#### Example Dosstring


```python
def add(value, num=0):
    """ Add a float to an integer

    Args:
        value (float): first number in the sum
        num (int): the integer to be added

    Returns:
        float: the sum of `value + num`

    >>> add(1., 2)
    3.0
    """
    pass
```


### Workflow

#### 1. Jira Ticket

Find or create a ticket describing the feature you are working on.
Assign the ticket to yourself.

#### 2. Branch off `develop`

Whenever you begin a new feature/task:

`git checkout develop -b feature/my-awesome-new-feature`

or

`git checkout develop -b feature/NSF-123-my-awesome-new-feature`

or

`git checkout develop -b bugfix/NSF-123-my-bug-fix`

#### 3. Write a docstring

Minimum docstring:

* 1-line description of what the new function/class/method/module does
* one doctest that will fail until you implement your new feature

Optionally (to aid your development):

* `Args:` see the Napoleon/Google/Numpy format example [above](https://github.com/aira/object_detector_app/blob/master/CONTRIBUTING.md#example-dosstring)

#### 4. Commit often

* Mention the Jira ticket number at the start of your commit message (when possible)
* Brief, active tense commit message.
* Transition your Jira Tickets at key whenever you can, to save yourself the Jira GUI shuffle.

When you're ready to commit for the first time (like after you added your docstring):

`git commit -am 'NSF-4 #start-progress add doctests for awesome new feature'`  

Send it to "#start-review" (the QA stage) rather than #done:


or 

`git commit -am 'NSF-4 #start-review finished adding color vectors'`  # when you are ready for someone to start code review

#### 5. Test

* Run your doctests.  
* Optionally do some manual testing on the command line
* Add additional doctests for edge cases based on your ipython hist: `%hist -o -p`

```bash
$ pytest -vs utils/
$ python -m pytest
$ python -m unittest discover -s object_detection -p "*_test.py"
```

#### 6. PR

* On github.com/aira issue a PR to the `develop` branch when you are ready for someone else on the team to run and review your changes.
* Ask someone to comment on your PR.
* Once your PR is Approved, all unittests and doctests pass, you or the reviewer may merge the PR and delete the feature branch

#### 7. Celebrate!

Brag about your *win* on the slack #ai-dev channel (or the #ai channel if you're really proud)

