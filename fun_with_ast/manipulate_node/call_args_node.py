import _ast


class CallArgs(_ast.stmt):
    """A node for handling arguments for a function call"""

    def __init__(self, args_list, keywords_list):
        self._fields = ['args', 'keywords']
        self.args = args_list
        self.keywords = keywords_list
