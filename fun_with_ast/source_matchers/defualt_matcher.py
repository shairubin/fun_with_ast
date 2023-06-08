import pprint

from fun_with_ast.placeholders.base_placeholder import Placeholder
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError, EmptyStackException
from fun_with_ast.source_matchers.base_matcher import SourceMatcher, MatchPlaceholderList, MatchPlaceholder, full_string
from fun_with_ast.placeholders.text import TextPlaceholder, EndParenMatcher, StartParenMatcher


class DefaultSourceMatcher(SourceMatcher):
    """Class to generate the source for a node."""

    def __init__(self, node, expected_parts, starting_parens=None, parent_node=None):
        super(DefaultSourceMatcher, self).__init__(node, starting_parens)
        previous_was_string = False
        # We validate that the expected parts does not contain two strings in
        # a row.
        for part in expected_parts:
            if not isinstance(part, Placeholder):
                raise ValueError('All expected parts must be Placeholder objects')
            if isinstance(part, TextPlaceholder) and not previous_was_string:
                previous_was_string = True
            # elif isinstance(part, TextPlaceholder) and previous_was_string:
            #     raise ValueError('Template cannot expect two strings in a row')
            else:
                previous_was_string = False
        self.expected_parts = expected_parts


    def _match(self, string):
        """Matches the string against self.expected_parts.

        Note that this is slightly peculiar in that it first matches fields,
        then goes back to match text before them. This is because currently we
        don't have matchers for every node, so by default, we separate each
        field with a '.*' TextSeparator, which is basically the current behavior
        of ast_annotate. This can change after we no longer have any need for
        '.*' TextSeparators.

        Args:
          string: {str} The string to match.

        Returns:
          The matched text.

        Raises:
          BadlySpecifiedTemplateError: If there is a mismatch between the
            expected_parts and the string.
          ValueError: If there is more than one TextPlaceholder in a rwo
        """
        remaining_string = self.MatchWhiteSpaces(string)
        remaining_string = self.MatchStartParens(remaining_string)


        try:
            remaining_string = MatchPlaceholderList(
                remaining_string, self.node, self.expected_parts,
                self.start_paren_matchers)
            remaining_string = self.MatchEndParen(remaining_string)
#            remaining_string = self.MatchWhiteSpaces(remaining_string, self.end_whitespace_matchers)

        except BadlySpecifiedTemplateError as e:
            raise BadlySpecifiedTemplateError(
                'When attempting to match string "{}" with {}, this '
                'error resulted:\n\n{}'
                    .format(string, self, e.message))
#        matched_string = self.GetSource()
        matched_string = DefaultSourceMatcher.GetSource(self)
        #if remaining_string :
        #    matched_string = string[:-len(remaining_string)]
        #leading_ws = self.GetWhiteSpaceText(self.start_whitespace_matchers)
        #start_parens = self.GetStartParenText()
        #end_parans = self.GetEndParenText()
        end_ws = self.GetWhiteSpaceText(self.end_whitespace_matchers)
#        result =  (leading_ws + start_parens + matched_string + end_parans + end_ws + self.end_of_line_comment)
#        result =  (matched_string + end_parans + end_ws + self.end_of_line_comment)
        result =  (matched_string + end_ws + self.end_of_line_comment)
        self.matched = True
        self.matched_source = result
        return result


    def GetSource(self):
        self.validated_call_to_match()
        if self.matched:
            return self.matched_source
        source_list = []
        for part in self.expected_parts:
            part_source = part.GetSource(self.node)
            source_list.append(part_source)
        source = ''.join(source_list)
        # if self.paren_wrapped:
        #     source = '{}{}{}'.format(
        #         self.GetStartParenText(),
        #         source,
        #         self.GetEndParenText())
        source = '{}{}{}'.format(
            self.GetStartParenText(),
            source,
            self.GetEndParenText())

        if self.start_whitespace_matchers:
            source = '{}{}'.format(self.GetWhiteSpaceText(self.start_whitespace_matchers), source)
        if self.end_whitespace_matchers:
            source = '{}{}'.format(source, self.GetWhiteSpaceText(self.end_whitespace_matchers))
        if self.end_of_line_comment:
            source = '{}{}'.format(source, self.end_of_line_comment)
        return source
    # def add_newline_to_source(self):
    #     part = self.expected_parts[-1]
    #     if isinstance(part, TextPlaceholder):
    #         if part.matched_text:
    #             part.matched_text += '\n'
    #         else:
    #             part.matched_text = '\n'
    #     else:
    #         raise NotImplementedError('Cannot add newline to non-text placeholder')

    # def MatchStartParens(self, string):
    #     """Matches the starting parens in a string."""
    #
    #     original_source_code =  full_string.get()
    #
    #     remaining_string = string
    #     #matched_parts = []
    #     try:
    #         while True:
    #             start_paren_matcher = StartParenMatcher()
    #             remaining_string = MatchPlaceholder(
    #                 remaining_string, None, start_paren_matcher)
    #             #self.start_paren_matchers.append(start_paren_matcher)
    #             #matched_parts.append(start_paren_matcher.matched_text)
    #             #node_name = str(self.node)
    #             self.parentheses_stack.push((start_paren_matcher, self))
    #     except BadlySpecifiedTemplateError:
    #         pass
    #     return remaining_string

    def __repr__(self):
        return ('DefaultSourceMatcher "{}" for node "{}" expecting to match "{}"'
                .format(super(DefaultSourceMatcher, self).__repr__(),
                        self.node,
                        pprint.pformat(self.expected_parts)))

    def MatchWhiteSpaces(self, remaining_string):
        ws_placeholder = self.start_whitespace_matchers[0]
        match_ws = ws_placeholder._match(None, remaining_string)
        remaining_string = remaining_string[len(match_ws):]
        #self.start_whitespace_matchers.append(ws_placeholder)
        return remaining_string

    # def MatchEndParen(self, string):
    #     """Matches the ending parens in a string."""
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
    #                 #if matcher_type[1] == str(self.node):
    #                 remaining_string = MatchPlaceholder( remaining_string, None, end_paren_matcher)
    #                 paired_matcher_info=   self.parentheses_stack.pop()
    #                 original_node_matcher = paired_matcher_info[1]
    #                 start_paren_matcher = paired_matcher_info[0]
    #                 self.end_paren_matchers.append(end_paren_matcher)
    #                 original_node_matcher.start_paren_matchers.append(start_paren_matcher)
    #             else:
    #                 break
    #                     #self.parentheses_stack.push((end_paren_matcher, str(self.node)))
    #             #self.paren_wrapped = True
    #     except BadlySpecifiedTemplateError:
    #         pass
    #     except EmptyStackException:
    #         pass
    #     if not remaining_string and len(self.start_paren_matchers)  > len(self.end_paren_matchers):
    #         raise BadlySpecifiedTemplateError('missing end paren at end of string')
    #     return remaining_string
#        new_end_matchers = []
#        new_start_matchers = []
#        min_size = min(len(self.start_paren_matchers), len(self.end_paren_matchers))
#        if min_size == 0:
#            return
#        for end_matcher in self.end_paren_matchers[:min_size]:
#            new_start_matchers.append(self.start_paren_matchers.pop())
#            new_end_matchers.append(end_matcher)
#        self.start_paren_matchers = new_start_matchers[::-1]
#        self.end_paren_matchers = new_end_matchers
