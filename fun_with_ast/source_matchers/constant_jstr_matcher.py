from fun_with_ast.source_matchers.constant_source_match import ConstantSourceMatcher


class ConstantJstrMatcher(ConstantSourceMatcher):
    def __init__(self, node, starting_parens=None, parent_node=None):
        ConstantSourceMatcher.__init__(self, node, starting_parens, parent_node)
        self.added_quote = '"'
    def _match(self, string):
        if string.find('"') != -1:
            self.added_quote = '\''
        string = self.added_quote + string + self.added_quote
        result = super(ConstantJstrMatcher, self)._match(string)
        result = self.GetSource()
        return result

    def GetSource(self):
        result = super(ConstantJstrMatcher, self).GetSource()
        if not (result.startswith('\'') or result.startswith('\"')):
            raise ValueError('ConstantJstrMatcher.GetSource does not start with \'')
        if not (result.endswith('\'') or result.endswith('\"')):
            raise ValueError('ConstantJstrMatcher.GetSource does not end with \'')
        result = result[1:-1]
        return result
