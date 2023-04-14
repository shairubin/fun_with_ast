import re

from  fun_with_ast.source_match import SourceMatcher, BadlySpecifiedTemplateError


class NumSourceMatcher(SourceMatcher):
    """Class to generate the source for an _ast.Num node."""

    def __init__(self, node, starting_parens=None):
        super(NumSourceMatcher, self).__init__(node, starting_parens)
        self.matched_num = None
        self.matched_as_str = None
        self.suffix = None

    def Match(self, string):
        remaining_string = self.MatchStartParens(string)
        comment_as_str, remaining_string = self.MatchCommentEOL(remaining_string, True)

        node_string_val = str(self.node.n)
        if isinstance(self.node.n, int):
            # Handle hex values
            if '0x' in string:
                raise NotImplementedError('not sporting hex value for ints')
#            num_as_str = re.match(r'(([ \t]*[+-]?\d+[ \t]*)((#*\S*)))', string)
            num_as_str = re.match(r'([ \t]*[+-]?\d+[ \t]*)((\)[ \t]*)*)', remaining_string)
            if not num_as_str:
                raise BadlySpecifiedTemplateError(
                    'String "{}" does not match Num pattern')
            int_as_str = num_as_str.group(1)
            end_parans = num_as_str.group(2)

        elif isinstance(self.node.n, float):
            int_as_str = re.match(r'[-+]?\d*.\d*', string).group(0)
        if int(int_as_str) != int(node_string_val):
            raise BadlySpecifiedTemplateError(
                'String "{}" should have started with string "{}"'
                .format(int_as_str, node_string_val))
        remaining_string = self.MatchEndParen(end_parans)

        self.matched_num = self.node.n
        start_parans_text = self.GetStartParenText()
        end_parans_text = self.GetEndParenText()
        self.matched_as_str = start_parans_text +  int_as_str + end_parans_text  + comment_as_str

#        unused_before, after = string.split(node_as_str, 1)
#        if after and after[0] in ('l', 'L', 'j', 'J'):
#            self.suffix = after[0]
#            node_as_str += after[0]
        return int_as_str

    def GetSource(self):
        node_as_str = str(self.node.n)
        if self.matched_num is not None and self.matched_num == self.node.n:
            node_as_str = self.matched_as_str
        if self.suffix:
            node_as_str += self.suffix
        return node_as_str

class BoolSourceMatcher(NumSourceMatcher):
    """Class to generate the source for an _ast.Num node."""

    def __init__(self, node, starting_parens=None):
        super(BoolSourceMatcher, self).__init__(node, starting_parens)
        self.matched_as_str = None

    def GetSource(self):
        node_as_str = str(self.node.n)
        if self.matched_bool is not None and self.matched_bool == self.node.n:
            node_as_str = self.matched_as_str
        if self.suffix:
            node_as_str += self.suffix
        return node_as_str

    def Match(self, string):
        remaining_string = self.MatchStartParens(string)
#        comment_as_str, remaining_string = self.MatchCommentEOL(remaining_string, True)

        node_string_val = str(self.node.n)
        if not isinstance(self.node.n, bool):
            raise BadlySpecifiedTemplateError('Node is not a bool')
        bool_match = re.match(r'(True|False)((\)[ \t]*)*)', remaining_string)
        if not bool_match:
            raise BadlySpecifiedTemplateError(
                'String "{}" does not match Bool pattern')
        bool_as_str = bool_match.group(1)
        end_parans = bool_match.group(2)
        self.MatchEndParen(end_parans)

        self.matched_bool = eval(bool_as_str)
        start_parans_text = self.GetStartParenText()
        end_parans_text = self.GetEndParenText()
        self.matched_as_str = start_parans_text +  bool_as_str + end_parans_text
        return bool_as_str
