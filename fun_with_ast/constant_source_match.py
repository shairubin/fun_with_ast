import ast

from fun_with_ast.num_source_match import NumSourceMatcher, BoolSourceMatcher
from fun_with_ast.source_matchers.str_source_match import StrSourceMatcher


class ConstantSourceMatcher():
    def __init__(self, node, starting_parens=None, parent_node=None):
        if not isinstance(node, ast.Constant):
            raise ValueError
        self.constant_node = node
        self.str_matcher = StrSourceMatcher(node, starting_parens)
        self.num_matcher = NumSourceMatcher(node, starting_parens)
        self.bool_matcher = BoolSourceMatcher(node, starting_parens)
        self.parent_node = parent_node

    def Match(self, string):
        if isinstance(self.constant_node.n, bool):
            return self.bool_matcher.Match(string)
        if isinstance(self.constant_node.n, int) and isinstance(self.constant_node.s, int):
            return self.num_matcher.Match(string)
        if isinstance(self.constant_node.n, str) and isinstance(self.constant_node.s, str):
            if isinstance(self.parent_node, ast.JoinedStr):
                raise NotImplementedError
            else:
                return self.str_matcher.Match(string)

    def GetSource(self):
        if isinstance(self.constant_node.n, bool) and isinstance(self.constant_node.s, int):
            return self.bool_matcher.GetSource()
        if isinstance(self.constant_node.n, int) and isinstance(self.constant_node.s, int):
            return self.num_matcher.GetSource()
        if isinstance(self.constant_node.n, str) and isinstance(self.constant_node.s, str):
            return self.str_matcher.GetSource()

        raise NotImplementedError
