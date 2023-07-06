from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError, EmptyStackException

from fun_with_ast.placeholders.base_match import MatchPlaceholder
from fun_with_ast.placeholders.text import StartParenMatcher, EndParenMatcher
from fun_with_ast.common_utils.stack import Stack


class ParanthesisStack(Stack):



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

        # original_source_code =  full_string.get()

        remaining_string = string
        # matched_parts = []
        try:
            while True:
                # for unused_i in range(len(self.start_paren_matchers)):
                end_paren_matcher = EndParenMatcher()
                # remaining_string = MatchPlaceholder(remaining_string, None, end_paren_matcher)
                matcher_type = self.parentheses_stack.peek()
                if isinstance(matcher_type[0], StartParenMatcher):
                    # if matcher_type[1] == str(self.node):
                    remaining_string = MatchPlaceholder(remaining_string, None, end_paren_matcher)
                    paired_matcher_info = self.parentheses_stack.pop()
                    original_node_matcher = paired_matcher_info[1]
                    start_paren_matcher = paired_matcher_info[0]
                    self.end_paren_matchers.append(end_paren_matcher)
                    original_node_matcher.start_paren_matchers.insert(0, start_paren_matcher)
                else:
                    break
        except BadlySpecifiedTemplateError:
            pass
        except EmptyStackException:
            pass
        if not remaining_string and len(self.start_paren_matchers) > len(self.end_paren_matchers):
            raise BadlySpecifiedTemplateError('missing end paren at end of string')
        return remaining_string

