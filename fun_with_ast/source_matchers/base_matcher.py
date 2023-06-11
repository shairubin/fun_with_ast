import re
from contextvars import ContextVar

from fun_with_ast.common_utils.stack import Stack
from fun_with_ast.placeholders.text import TextPlaceholder, StartParenMatcher, EndParenMatcher
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError, EmptyStackException
from fun_with_ast.placeholders.node import ValidateStart
from fun_with_ast.placeholders.string_parser import StripStartParens
from fun_with_ast.placeholders.whitespace import WhiteSpaceTextPlaceholder

full_string = ContextVar('full_string')
def MatchPlaceholder(string, node, placeholder):
    """Match a placeholder against a string."""
    matched_text = placeholder._match(node, string)
    if not matched_text:
        return string
    ValidateStart(string, matched_text)
    if not isinstance(placeholder, TextPlaceholder):
        matched_text = StripStartParens(matched_text)
    before, after = string.split(matched_text, 1)
    if StripStartParens(before):
        raise BadlySpecifiedTemplateError(
            'string "{}" should have started with placeholder "{}"'
                .format(string, placeholder))
    return after

def MatchPlaceholderList(string, node, placeholders, starting_parens=None):
    remaining_string = string
    for placeholder in placeholders:
        if remaining_string == string:
            placeholder.SetStartingParens(starting_parens)
        remaining_string = MatchPlaceholder(
            remaining_string, node, placeholder)
    return remaining_string

class SourceMatcher(object):
    """Base class for all SourceMatcher objects.

    These are designed to match the source that corresponds to a given node.
    """
    parentheses_stack = Stack()
    def __init__(self, node, stripped_parens=None):
        self.node = node
        self.end_paren_matchers = []
        self.start_whitespace_matchers =  [WhiteSpaceTextPlaceholder()]
        self.end_whitespace_matchers =  [WhiteSpaceTextPlaceholder()]
        self.paren_wrapped = False
        self.end_of_line_comment = ''
        if not stripped_parens:
            stripped_parens = []
        self.start_paren_matchers = stripped_parens
        self.matched = False
        self.matched_source = None
        self.parentheses_stack_depth = self.parentheses_stack.size


    def do_match(self, string):
        self._check_balance_parentheses(string)
        SourceMatcher.parentheses_stack.reset()
        result = self._match(string)
        return result


    def _match(self, string):
        #SourceMatcher.parentheses_stack.push(self)
        raise NotImplementedError

    def GetSource(self):
        raise NotImplementedError


    def FixIndentation(self, new_ident):
        """Fix the indentation of the source."""
        current_ws = self.start_whitespace_matchers[0].matched_text
        if current_ws:
            current_ident = len(self.start_whitespace_matchers[0].matched_text)
            if current_ident == new_ident:
                return
        else:
            self.start_whitespace_matchers[0].matched_text = ' ' * new_ident

    # def MatchStartParens(self, string):
    #     """Matches the starting parens in a string."""
    #     original_source_code =  full_string.get()
    #
    #     remaining_string = string
    #     matched_parts = []
    #     try:
    #         while True:
    #             start_paren_matcher = StartParenMatcher()
    #             remaining_string = MatchPlaceholder(
    #                 remaining_string, None, start_paren_matcher)
    #             self.start_paren_matchers.append(start_paren_matcher)
    #             matched_parts.append(start_paren_matcher.matched_text)
    #             self.parentheses_stack.push(start_paren_matcher)
    #     except BadlySpecifiedTemplateError:
    #         pass
    #     return remaining_string

    def MatchStartParens(self, string):
        """Matches the starting parens in a string."""

        #original_source_code =  full_string.get()

        remaining_string = string
        #matched_parts = []
        try:
            while True:
                start_paren_matcher = StartParenMatcher()
                remaining_string = MatchPlaceholder(
                    remaining_string, None, start_paren_matcher)
                #self.start_paren_matchers.append(start_paren_matcher)
                #matched_parts.append(start_paren_matcher.matched_text)
                #node_name = str(self.node)
                self.parentheses_stack.push((start_paren_matcher, self))
        except BadlySpecifiedTemplateError:
            pass
        return remaining_string

    # def MatchEndParen(self, string):
    #     """Matches the ending parens in a string."""
    #
    #     if not self.start_paren_matchers:
    #         return
    #     remaining_string = string
    #     matched_parts = []
    #     try:
    #         while True:
    #         #for unused_i in range(len(self.start_paren_matchers)):
    #             end_paren_matcher = EndParenMatcher()
    #             remaining_string = MatchPlaceholder(
    #                 remaining_string, None, end_paren_matcher)
    #             self.end_paren_matchers.append(end_paren_matcher)
    #             matched_parts.append(end_paren_matcher.matched_text)
    #             self.paren_wrapped = True
    #             if isinstance(self.parentheses_stack.peek(), StartParenMatcher):
    #                 self.parentheses_stack.pop()
    #             else:
    #                 self.parentheses_stack.push(end_paren_matcher)
    #     except BadlySpecifiedTemplateError:
    #         pass
    #     except EmptyStackException:
    #         #raise BadlySpecifiedTemplateError('unmatched end paren')
    #         raise
    #     if not remaining_string and len(self.start_paren_matchers)  > len(self.end_paren_matchers):
    #         raise BadlySpecifiedTemplateError('missing end paren at end of string')
    #
    #     new_end_matchers = []
    #     new_start_matchers = []
    #     min_size = min(len(self.start_paren_matchers), len(self.end_paren_matchers))
    #     if min_size == 0:
    #         return
    #     for end_matcher in self.end_paren_matchers[:min_size]:
    #         new_start_matchers.append(self.start_paren_matchers.pop())
    #         new_end_matchers.append(end_matcher)
    #     self.start_paren_matchers = new_start_matchers[::-1]
    #     self.end_paren_matchers = new_end_matchers

    def MatchEndParen(self, string):
        """Matches the ending parens in a string."""

        end_paren_matcher = EndParenMatcher()
        try:
            MatchPlaceholder(string, None, end_paren_matcher)
        except BadlySpecifiedTemplateError:
            return

        #original_source_code =  full_string.get()

        remaining_string = string
        #matched_parts = []
        try:
            while True :
            #for unused_i in range(len(self.start_paren_matchers)):
                end_paren_matcher = EndParenMatcher()
                #remaining_string = MatchPlaceholder(remaining_string, None, end_paren_matcher)
                matcher_type = self.parentheses_stack.peek()
                if isinstance(matcher_type[0], StartParenMatcher):
                    #if matcher_type[1] == str(self.node):
                    remaining_string = MatchPlaceholder(remaining_string, None, end_paren_matcher)
                    paired_matcher_info = self.parentheses_stack.pop()
                    original_node_matcher = paired_matcher_info[1]
                    start_paren_matcher = paired_matcher_info[0]
                    self.end_paren_matchers.append(end_paren_matcher)
                    original_node_matcher.start_paren_matchers.insert(0,start_paren_matcher)
                else:
                    break
                        #self.parentheses_stack.push((end_paren_matcher, str(self.node)))
                #self.paren_wrapped = True
        except BadlySpecifiedTemplateError:
            pass
        except EmptyStackException:
            pass
            #raise EmptyStackException('unmatched end paren')
        if not remaining_string and len(self.start_paren_matchers)  > len(self.end_paren_matchers):
            raise BadlySpecifiedTemplateError('missing end paren at end of string')
        return remaining_string

    # nice example for creating unit test
    def MatchCommentEOL(self, string, remove_comment=False):
        if string is None:
            return ''
        remaining_string = string
        comment = ''

        full_line  = re.match(r'(.*)(#.*)', string)
        if full_line:
            comment = full_line.group(2)
        if comment:
            self.end_of_line_comment = comment
        if remove_comment and full_line:
            remaining_string = full_line.group(1)
        return comment

    # def MatchWhiteSpaces(self, string, in_matcher):
    #     """Matches the  whitespaces  in a string."""
    #     remaining_string = string
    #     matched_parts = []
    #     try:
    #         ws_matcher = GetWhiteSpaceMatcher()
    #         remaining_string = MatchPlaceholder(
    #                 remaining_string, None, ws_matcher)
    #         if remaining_string != string:
    #             in_matcher.append(ws_matcher)
    #             matched_parts.append(ws_matcher._matched_text)
    #     except BadlySpecifiedTemplateError:
    #         pass
    #     return remaining_string


    # def MatchStartLeadingWhiteSpaces(self, string):
    #     """Matches the starting whitespaces  in a string."""
    #     remaining_string = string
    #     matched_parts = []
    #     try:
    #         start_ws_matcher = GetWhiteSpaceMatcher()
    #         remaining_string = MatchPlaceholder(
    #                 remaining_string, None, start_ws_matcher)
    #         if remaining_string != string:
    #             self.start_whitespace_matchers.append(start_ws_matcher)
    #             matched_parts.append(start_ws_matcher._matched_text)
    #     except BadlySpecifiedTemplateError:
    #         pass
    #     return remaining_string


    def GetStartParenText(self):
        result = ''
        # if self.paren_wrapped:
        #     for matcher in self.start_paren_matchers:
        #         result += matcher.GetSource(None)
#        if self.paren_wrapped:
        for matcher in self.start_paren_matchers:
            result += matcher.GetSource(None)
        return result

    def GetWhiteSpaceText(self, in_matcher):
        result = ''
        if in_matcher:
            result = ''.join(matcher.GetSource(None)
                           for matcher in in_matcher)
        return result

    def GetEndParenText(self):
#        if self.paren_wrapped:
        return ''.join(matcher.GetSource(None)
                       for matcher in self.end_paren_matchers)
        return ''

    def add_newline_to_source(self):
        part = self.expected_parts[-1]
        if isinstance(part, TextPlaceholder):
            if part.matched_text:
                part.matched_text += '\n'
            else:
                part.matched_text = '\n'
        else:
            raise NotImplementedError('Cannot add newline to non-text placeholder')
        self.matched_source = None
        self.matched = False
    def validated_call_to_match(self):
        if self.matched and self.matched_source is None:
            raise ValueError('Internal Error: matched_text must be set if is_matched is True')
        if not self.matched and self.matched_source is not None:
            raise ValueError('Internal Error: matched_text must be None if is_matched is False')
        if hasattr(self, 'matched_text'):
            raise ValueError('Internal Error: dont have this field - ever')

    def _check_balance_parentheses(self, string):
        left = right = 0
        for c in string:
            if c == '(':
                left += 1
            elif c == ')':
                right += 1

        if left != right:
            raise BadlySpecifiedTemplateError('unbalanced parentheses')



