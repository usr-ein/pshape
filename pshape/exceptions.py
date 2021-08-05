class InteractiveSourceError(Exception):
    """The only way that the inspect module can display source code is if the code came from a file that it can access.
    Source typed at an interactive prompt is discarded as soon as it is parsed, there's simply no way for inspect to access it. – @jasonharper"""

    pass


class ParsingError(ValueError):
    """Occurs when we can't parse argument names using inspect"""

    pass
