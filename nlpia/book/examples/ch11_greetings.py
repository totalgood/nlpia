""" A naive way to build a Finit State Machine to extract piece of information """


def find_greeting(s):
    """ Return the the greeting string Hi, Hello, or Yo if it occurs at the beginning of a string

    >>> find_greeting('Hi Mr. Turing!')
    'Hi'
    >>> find_greeting('Hello, Rosa.')
    'Hello'
    >>> find_greeting("Yo, what's up?")
    'Yo'
    >>> find_greeting("Hello")
    'Hello'
    >>> print(find_greeting("hello"))
    None
    >>> print(find_greeting("HelloWorld"))
    None
    """
    if s[0] == 'H':
        if s[:3] in ['Hi', 'Hi ', 'Hi,', 'Hi!']:
            return s[:2]
        elif s[:6] in ['Hello', 'Hello ', 'Hello,', 'Hello!']:
            return s[:5]
    elif s[0] == 'Y':
        if s[1] == 'o' and s[:3] in ['Yo', 'Yo,', 'Yo ', 'Yo!']:
            return s[:2]
    return None
