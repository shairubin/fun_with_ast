from fun_with_ast.defualt_source_matcher_source_match import DefaultSourceMatcher
from fun_with_ast.list_placeholder_source_match import ListFieldPlaceholder
#from fun_with_ast.source_matchers.str_source_match import StrSourceMatcher
#from fun_with_ast.string_part_placeholder import JoinedStringPartPlaceholder
#from fun_with_ast.source_matcher_source_match import MatchPlaceholder
from fun_with_ast.text_placeholder_source_match import TextPlaceholder
from fun_with_ast.utils_source_match import _FindQuoteEnd


class JoinedStrSourceMatcher(DefaultSourceMatcher):
    """Source matcher for _ast.Tuple nodes."""

    def __init__(self, node, starting_parens=None):
        expected_parts = [
            TextPlaceholder(r'f', 'f'),
            ListFieldPlaceholder(r'values')
#            TextPlaceholder(r'\'', '\'')
        ]
        super(JoinedStrSourceMatcher, self).__init__(
            node, expected_parts, starting_parens)

    def Match(self, string):
#        without_end_quote = _FindQuoteEnd(string[2:], "'")
        #string = string[:-1]
        matched_text = super(JoinedStrSourceMatcher, self).Match(string)
        return matched_text

    def MatchStartParens(self, remaining_string):
        return remaining_string


# class JoinedStrSourceMatcher(StrSourceMatcher):
#     def __init__(self, node, starting_parens=None):
#         super(JoinedStrSourceMatcher, self).__init__(node, starting_parens)
#         self.value_placeholder = ListFieldPlaceholder('values')
#
#     def _get_original_string(self):
#         self.original_s = ''
#
#     def _part_place_holder(self):
#         return JoinedStringPartPlaceholder()
#
#     def Match(self, string):
#         part = self._part_place_holder()
#         remaining_string = MatchPlaceholder(string, None, part)
#         self.quote_parts.append(part)
#
#     def _handle_multipart(self, remaining_string):
#         pass