from fun_with_ast.source_matchers.defualt_matcher import DefaultSourceMatcher
from fun_with_ast.placeholders.list_placeholder import SeparatedListFieldPlaceholder
from fun_with_ast.placeholders.text import TextPlaceholder


class TupleSourceMatcher(DefaultSourceMatcher):
    pass
#
#     def __init__(self, node, starting_parens=None, parent=None):
#         expected_parts = [
#             TextPlaceholder(r'(\s*\(|\s*)', ''),
#             SeparatedListFieldPlaceholder(
#                 'elts', before_separator_placeholder=TextPlaceholder(r'[ \t]*,[ \t]*', ',')),
#             TextPlaceholder(r'(\s*,?\s*\)|\s*)[ \t]*(#\S*)*', ')')
#         ]
#         super(TupleSourceMatcher, self).__init__(
#             node, expected_parts, starting_parens)
#
#     def Match(self, string):
#         matched_text = super(TupleSourceMatcher, self).Match(string)
#         return matched_text
# #        if not self.paren_wrapped:
# #            matched_text = matched_text.rstrip()
# #            return super(TupleSourceMatcher, self).Match(matched_text)
#
#     def MatchStartParens(self, remaining_string):
#         #raise NotImplementedError('use main stack')
#         return remaining_string
#         # if remaining_string.startswith('(('):
#         #    raise NotImplementedError('Currently not supported')
#         # if remaining_string.startswith('('):
#         #    return remaining_string
#         # raise ValueError('Tuple does not start with (')
