from fun_with_ast.defualt_source_matcher_source_match import DefaultSourceMatcher
from fun_with_ast.list_placeholder_source_match import SeparatedListFieldPlaceholder
from fun_with_ast.text_placeholder_source_match import TextPlaceholder


class TupleSourceMatcher(DefaultSourceMatcher):
    """Source matcher for _ast.Tuple nodes."""

    def __init__(self, node, starting_parens=None):
        expected_parts = [
            TextPlaceholder(r'\s*\(', ''),
            SeparatedListFieldPlaceholder(
                'elts', before_separator_placeholder=TextPlaceholder(r'[ \t]*,[ \t]*', ',')),
            TextPlaceholder(r'\s*,?\s*\)[ \t]*(#\S*)*', ')')
        ]
        super(TupleSourceMatcher, self).__init__(
            node, expected_parts, starting_parens)

    def Match(self, string):
        matched_text = super(TupleSourceMatcher, self).Match(string)
        return matched_text
#        if not self.paren_wrapped:
#            matched_text = matched_text.rstrip()
#            return super(TupleSourceMatcher, self).Match(matched_text)

    def MatchStartParens(self, remaining_string):
        return remaining_string
        # if remaining_string.startswith('(('):
        #    raise NotImplementedError('Currently not supported')
        # if remaining_string.startswith('('):
        #    return remaining_string
        # raise ValueError('Tuple does not start with (')