import ast

from fun_with_ast.manipulate_node.call_args_node import CallArgs
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError, EmptyStackException, \
    ReachedEndOfNodeException

from fun_with_ast.placeholders.base_match import MatchPlaceholder
from fun_with_ast.placeholders.text import StartParenMatcher, EndParenMatcher
from fun_with_ast.common_utils.stack import Stack


class ParanthesisStack(Stack):



    def MatchStartParens(self, matcher, string):
        """Matches the starting parens in a string."""
        remaining_string = string
        try:
            while True:
                start_paren_matcher = StartParenMatcher()
                remaining_string = MatchPlaceholder(
                    remaining_string, None, start_paren_matcher)
                self.push((start_paren_matcher, matcher))
        except BadlySpecifiedTemplateError:
            pass
        return remaining_string


    def MatchEndParens(self, matcher, string):
        """Matches the ending parens in a string."""

        end_paren_matcher = EndParenMatcher()
        try:
            MatchPlaceholder(string, None, end_paren_matcher)
        except BadlySpecifiedTemplateError:
            return


        remaining_string = string
        try:
            while True:
                # for unused_i in range(len(self.start_paren_matchers)):
                end_paren_matcher = EndParenMatcher()
                orig_start_paren_matcher, orig_source_matcher = self.peek()
                if matcher is not orig_source_matcher:
                    if matcher.node.parent_node is not None:
                        parent_matcher = matcher.node.parent_node.node_matcher
                        if orig_source_matcher is not parent_matcher:
                            pass
                            #raise ReachedEndOfNodeException(matcher, orig_source_matcher, remaining_string)
                        if orig_source_matcher is parent_matcher and parent_matcher.node.parent_node is not None:
                            pass
                            #raise ReachedEndOfNodeException(matcher, orig_source_matcher, remaining_string)

                    else:
                        pass # unclear what to do in this case
                if isinstance(orig_start_paren_matcher, StartParenMatcher):
                    remaining_string = MatchPlaceholder(remaining_string, None, end_paren_matcher)
                    original_node_matcher = orig_source_matcher
                    start_paren_matcher = orig_start_paren_matcher
                    self.pop()
                    matcher.end_paren_matchers.append(end_paren_matcher)
                    original_node_matcher.start_paren_matchers.insert(0, start_paren_matcher)
                else:
                    ValueError('parentheses stack is not balanced')
        except BadlySpecifiedTemplateError:
            pass
        except EmptyStackException:
            pass
        if not remaining_string and len(matcher.start_paren_matchers) > len(matcher.end_paren_matchers):
            raise BadlySpecifiedTemplateError('missing end paren at end of string')
        return remaining_string

