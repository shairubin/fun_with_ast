from fun_with_ast.get_source import GetSource
from fun_with_ast.source_matchers.base_matcher import SourceMatcher, MatchPlaceholder, full_string
from fun_with_ast.placeholders.string_parser import StringParser
from fun_with_ast.placeholders.text import TextPlaceholder, EndParenMatcher, StartParenMatcher
from fun_with_ast.common_utils.utils_source_match import _GetListDefault
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError, EmptyStackException


class BoolOpSourceMatcher(SourceMatcher):
    """Class to generate the source for an _ast.BoolOp node."""

    def __init__(self, node, starting_parens=None, parent=None):
        super(BoolOpSourceMatcher, self).__init__(node, starting_parens)
        self.separator_placeholder = TextPlaceholder(r'\s*', ' ')
        self.matched_placeholders = []

    def GetSeparatorCopy(self):
        copy = self.separator_placeholder.Copy()
        self.matched_placeholders.append(copy)
        return copy

    def _match(self, string):
        remaining_string = self.MatchStartParens(string)

        elements = [self.node.values[0]]
        for value in self.node.values[1:]:
            elements.append(self.GetSeparatorCopy())
            elements.append(self.node.op)
            elements.append(self.GetSeparatorCopy())
            elements.append(value)

#        parser = StringParser(remaining_string, elements, self.start_paren_matchers)
        parser = StringParser(remaining_string, elements)
        matched_text = ''.join(parser.matched_substrings)
        remaining_string = parser.remaining_string

        self.MatchEndParen(remaining_string)

#        result =  self.GetStartParenText() + matched_text + self.GetEndParenText()
        result =  BoolOpSourceMatcher.GetSource(self)
        return result
    def GetSource(self):
        source_list = []
        # if self.paren_wrapped:
        #     source_list.append(self.GetStartParenText())
        source_list.append(self.GetStartParenText())

        source_list.append(GetSource(self.node.values[0]))
        index = 0
        for value in self.node.values[1:]:
            source_list.append(_GetListDefault(
                self.matched_placeholders,
                index,
                self.separator_placeholder).GetSource(None))
            source_list.append(GetSource(self.node.op))
            index += 1
            source_list.append(_GetListDefault(
                self.matched_placeholders,
                index,
                self.separator_placeholder).GetSource(None))
            source_list.append(GetSource(value))
            index += 1
        #if self.paren_wrapped:
        source_list.append(self.GetEndParenText())
        return ''.join(source_list)
    # def MatchEndParen(self, string):  #TODO: refactor this - return to base matcher class
    #     """Matches the ending parens in a string."""
    #
    #
    #     end_paren_matcher = EndParenMatcher()
    #     try:
    #         MatchPlaceholder(string, None, end_paren_matcher)
    #     except BadlySpecifiedTemplateError:
    #         return
    #
    #     original_source_code =  full_string.get()
    #
    #     remaining_string = string
    #     #matched_parts = []
    #     try:
    #         while True:
    #         #for unused_i in range(len(self.start_paren_matchers)):
    #             end_paren_matcher = EndParenMatcher()
    #             matcher_type = self.parentheses_stack.peek()
    #             if isinstance(matcher_type[0], StartParenMatcher):
    #                 if matcher_type[1] == str(self.node):
    #                     remaining_string = MatchPlaceholder( remaining_string, None, end_paren_matcher)
    #                     start_paren_matcher =  self.parentheses_stack.pop()[0]
    #                     self.end_paren_matchers.append(end_paren_matcher)
    #                     self.start_paren_matchers.append(start_paren_matcher)
    #
    #                 else:
    #                     break
    #                     #self.parentheses_stack.push((end_paren_matcher, str(self.node)))
    #             #self.paren_wrapped = True
    #     except BadlySpecifiedTemplateError:
    #         pass
    #     except EmptyStackException:
    #         pass
    #     if not remaining_string and len(self.start_paren_matchers)  > len(self.end_paren_matchers):
    #         raise BadlySpecifiedTemplateError('missing end paren at end of string')
    #     return remaining_string




        #
        # if not self.start_paren_matchers:
        #     return
        # remaining_string = string
        # matched_parts = []
        # try:
        #     while True:
        #     #for unused_i in range(len(self.start_paren_matchers)):
        #         end_paren_matcher = EndParenMatcher()
        #         remaining_string = MatchPlaceholder(
        #             remaining_string, None, end_paren_matcher)
        #         self.end_paren_matchers.append(end_paren_matcher)
        #         matched_parts.append(end_paren_matcher.matched_text)
        #         self.paren_wrapped = True
        #         if isinstance(self.parentheses_stack.peek(), StartParenMatcher):
        #             self.parentheses_stack.pop()
        #         else:
        #             self.parentheses_stack.push(end_paren_matcher)
        # except BadlySpecifiedTemplateError:
        #     pass
        # except EmptyStackException:
        #     #raise BadlySpecifiedTemplateError('unmatched end paren')
        #     raise
        # if not remaining_string and len(self.start_paren_matchers)  > len(self.end_paren_matchers):
        #     raise BadlySpecifiedTemplateError('missing end paren at end of string')
        #
        # new_end_matchers = []
        # new_start_matchers = []
        # min_size = min(len(self.start_paren_matchers), len(self.end_paren_matchers))
        # if min_size == 0:
        #     return
        # for end_matcher in self.end_paren_matchers[:min_size]:
        #     new_start_matchers.append(self.start_paren_matchers.pop())
        #     new_end_matchers.append(end_matcher)
        # self.start_paren_matchers = new_start_matchers[::-1]
        # self.end_paren_matchers = new_end_matchers
#     def MatchStartParens(self, string):
#         """Matches the starting parens in a string."""
#
#         original_source_code =  full_string.get()
#
#         remaining_string = string
#         #matched_parts = []
#         try:
#             while True:
#                 start_paren_matcher = StartParenMatcher()
#                 remaining_string = MatchPlaceholder(
#                     remaining_string, None, start_paren_matcher)
# #                MatchPlaceholder(remaining_string, None, start_paren_matcher)
#                 #self.start_paren_matchers.append(start_paren_matcher)
#                 #matched_parts.append(start_paren_matcher.matched_text)
#                 #node_name = str(self.node)
#                 self.parentheses_stack.push((start_paren_matcher, self))
#         except BadlySpecifiedTemplateError:
#             pass
#         return remaining_string # TODO: do not fix this?
