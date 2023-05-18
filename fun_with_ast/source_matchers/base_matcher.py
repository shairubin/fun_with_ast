import re

from fun_with_ast.placeholders.text_placeholder import TextPlaceholder, GetStartParenMatcher, GetEndParenMatcher
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.placeholders.node import ValidateStart
from fun_with_ast.placeholders.string_parser import StripStartParens
from fun_with_ast.placeholders.whitespace_placeholder import WhiteSpaceTextPlaceholder


def MatchPlaceholder(string, node, placeholder):
    """Match a placeholder against a string."""
    matched_text = placeholder.Match(node, string)
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


    def Match(self, string):
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
        matched_parts = []
        try:
            while True:
                start_paren_matcher = GetStartParenMatcher()
                remaining_string = MatchPlaceholder(
                    remaining_string, None, start_paren_matcher)
                self.start_paren_matchers.append(start_paren_matcher)
                matched_parts.append(start_paren_matcher.matched_text)
        except BadlySpecifiedTemplateError:
            pass
        return remaining_string

    def MatchEndParen(self, string):
        """Matches the ending parens in a string."""
        if not self.start_paren_matchers:
            return
        remaining_string = string
        matched_parts = []
        try:
            while True:
            #for unused_i in range(len(self.start_paren_matchers)):
                end_paren_matcher = GetEndParenMatcher()
                remaining_string = MatchPlaceholder(
                    remaining_string, None, end_paren_matcher)
                self.end_paren_matchers.append(end_paren_matcher)
                matched_parts.append(end_paren_matcher.matched_text)
                self.paren_wrapped = True
        except BadlySpecifiedTemplateError:
            pass

        if not remaining_string and len(self.start_paren_matchers) != len(self.end_paren_matchers):
            raise BadlySpecifiedTemplateError('string should have ended with paren')

        new_end_matchers = []
        new_start_matchers = []
        min_size = min(len(self.start_paren_matchers), len(self.end_paren_matchers))
        if min_size == 0:
            return
        for end_matcher in self.end_paren_matchers[:min_size]:
            new_start_matchers.append(self.start_paren_matchers.pop())
            new_end_matchers.append(end_matcher)
        self.start_paren_matchers = new_start_matchers[::-1]
        self.end_paren_matchers = new_end_matchers

    # nice example for creating unit test
    def MatchCommentEOL(self, string, remove_comment=False):
        remaining_string = string
        comment = ''
        full_line  = re.match(r'(.*)(#.*)', string)
        if full_line:
            comment = full_line.group(2)
        if comment:
            self.end_of_line_comment = comment
        if remove_comment and full_line:
            remaining_string = full_line.group(1)
        return comment, remaining_string

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
    #             matched_parts.append(ws_matcher.matched_text)
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
    #             matched_parts.append(start_ws_matcher.matched_text)
    #     except BadlySpecifiedTemplateError:
    #         pass
    #     return remaining_string


    def GetStartParenText(self):
        result = ''
        if self.paren_wrapped:
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
        if self.paren_wrapped:
            return ''.join(matcher.GetSource(None)
                           for matcher in self.end_paren_matchers)
        return ''
