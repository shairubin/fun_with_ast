

from fun_with_ast.placeholders.composite import FieldPlaceholder
from fun_with_ast.source_matchers.defualt_matcher import DefaultSourceMatcher
from fun_with_ast.placeholders.text import TextPlaceholder
from fun_with_ast.placeholders.whitespace import WhiteSpaceTextPlaceholder


class SyntaxFreeLineMatcher(DefaultSourceMatcher):
    """Class to generate the source for a node."""

    def __init__(self, node, expected_parts, starting_parens=None, parent_node=None):
        parts = [FieldPlaceholder('full_line'), TextPlaceholder(r'\n', '\n')]
        super(SyntaxFreeLineMatcher, self).__init__(node, parts, starting_parens)

    def MatchWhiteSpaces(self, remaining_string):
        ws_placeholder = WhiteSpaceTextPlaceholder()
        self.start_whitespace_matchers.append(ws_placeholder)
        return remaining_string # NOTE: we return remaining_string !
