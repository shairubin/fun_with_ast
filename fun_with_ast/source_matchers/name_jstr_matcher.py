import ast
from types import NoneType

from fun_with_ast.source_matchers.defualt_matcher import DefaultSourceMatcher

from fun_with_ast.source_matchers.constant_source_match import ConstantSourceMatcher


class NameJstrMatcher(DefaultSourceMatcher):
    def __init__(self, node, starting_parens=None, parent_node=None):
        DefaultSourceMatcher.__init__(self, node, starting_parens, parent_node)

    def _match(self, string):
        result = super(NameJstrMatcher, self)._match(string)
        return result

    def GetSource(self):
        raise NotImplementedError('Do not use NameJstrMatcher.GetSource')
        result = super(ConstantJstrMatcher, self).GetSource()
        if not (result.startswith('\'') or result.startswith('\"')):
            raise ValueError('ConstantJstrMatcher.GetSource does not start with \'')
        if not (result.endswith('\'') or result.endswith('\"')):
            raise ValueError('ConstantJstrMatcher.GetSource does not end with \'')
        result = result[1:-1]
        return result
