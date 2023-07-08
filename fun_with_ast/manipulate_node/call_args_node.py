import _ast


class CallArgs(_ast.stmt):
    """Class defining a new node that has no syntax (only optional comments)."""

    def __init__(self, args_list):
        self._fields = ['args']
        self.args = args_list
