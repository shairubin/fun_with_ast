import ast
from types import NoneType

from fun_with_ast.source_matchers.constant_source_match import ConstantSourceMatcher


class ConstantJstrMatcher(ConstantSourceMatcher):
    def __init__(self, node, starting_parens=None, parent_node=None):
        ConstantSourceMatcher.__init__(self, node, starting_parens, parent_node)

    def _match(self, string):
#        string = self.node.default_quote + string + self.node.default_quote
        string = '\"' + string + '\"'
        result = super(ConstantJstrMatcher, self)._match(string)
        result = self.GetSource()
        return result
        #raise NotImplementedError('Do not use ConstantJstrMatcher._match')

    def GetSource(self):
        result = super(ConstantJstrMatcher, self).GetSource()
        if not (result.startswith('\'') or result.startswith('\"')):
            raise ValueError('ConstantJstrMatcher.GetSource does not start with \'')
        if not (result.endswith('\'') or result.endswith('\"')):
            raise ValueError('ConstantJstrMatcher.GetSource does not end with \'')
        result = result[1:-1]
        return result
