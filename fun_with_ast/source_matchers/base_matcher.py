import ast
import re
from contextvars import ContextVar

from fun_with_ast.common_utils.parenthese_stack import ParanthesisStack
from fun_with_ast.placeholders.base_match import MatchPlaceholder
from fun_with_ast.placeholders.text import TextPlaceholder, StartParenMatcher, EndParenMatcher
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError, EmptyStackException
from fun_with_ast.placeholders.whitespace import WhiteSpaceTextPlaceholder

full_string = ContextVar('full_string')


class SourceMatcher(object):
    """Base class for all SourceMatcher objects.

    These are designed to match the source that corresponds to a given node.
    """
    parentheses_stack = ParanthesisStack()
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
        if len(result) < len(string):
            try:
                ast.parse(string)
            except SyntaxError:
                raise BadlySpecifiedTemplateError(f'string {string} is not valid python')
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


    def MatchStartParens(self, string):
        """Matches the starting parens in a string."""


        remaining_string = string
        try:
            while True:
                start_paren_matcher = StartParenMatcher()
                remaining_string = MatchPlaceholder(
                    remaining_string, None, start_paren_matcher)
                self.parentheses_stack.push((start_paren_matcher, self))
        except BadlySpecifiedTemplateError:
            pass
        return remaining_string


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
        except BadlySpecifiedTemplateError:
            pass
        except EmptyStackException:
            pass
        if not remaining_string and len(self.start_paren_matchers)  > len(self.end_paren_matchers):
            raise BadlySpecifiedTemplateError('missing end paren at end of string')
        return remaining_string

    # nice example for creating unit test
    def MatchCommentEOL(self, string, remove_comment=False):
        if string is None:
            return ''
        remaining_string = string
        comment = ''

        full_line  = re.match(r'(\s*)(#.*)', string)
        if full_line:
            comment = full_line.group(2)
        if comment:
            self.end_of_line_comment = comment
        if remove_comment and full_line:
            remaining_string = full_line.group(1)
        return comment



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



