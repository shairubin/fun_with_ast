import pprint

from fun_with_ast.placeholders.base_placeholder import Placeholder
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matcher_source_match import SourceMatcher, MatchPlaceholderList
from fun_with_ast.placeholders.text_placeholder import TextPlaceholder


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
            elif isinstance(part, TextPlaceholder) and previous_was_string:
                raise ValueError('Template cannot expect two strings in a row')
            else:
                previous_was_string = False
        self.expected_parts = expected_parts
        self.matched = False

    def Match(self, string):
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
#        _ , remaining_string = self.MatchCommentEOL(string, True)
        remaining_string = self.MatchStartParens(string)

        try:
            remaining_string = MatchPlaceholderList(
                remaining_string, self.node, self.expected_parts,
                self.start_paren_matchers)
            self.MatchEndParen(remaining_string)
#            remaining_string = self.MatchWhiteSpaces(remaining_string, self.end_whitespace_matchers)

        except BadlySpecifiedTemplateError as e:
            raise BadlySpecifiedTemplateError(
                'When attempting to match string "{}" with {}, this '
                'error resulted:\n\n{}'
                    .format(string, self, e.message))
        matched_string = string
        if remaining_string:
            matched_string = string[:-len(remaining_string)]
        leading_ws = self.GetWhiteSpaceText(self.start_whitespace_matchers)
        start_parens = self.GetStartParenText()
        end_parans = self.GetEndParenText()
        end_ws = self.GetWhiteSpaceText(self.end_whitespace_matchers)
        result =  (leading_ws + start_parens + matched_string + end_parans + end_ws + self.end_of_line_comment)
        return result


    def GetSource(self):
        source_list = []
        for part in self.expected_parts:
            source_list.append(part.GetSource(self.node))
        source = ''.join(source_list)
        if self.paren_wrapped:
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
    def add_newline_to_source(self):
        part = self.expected_parts[-1]
        if isinstance(part, TextPlaceholder):
            part.matched_text += '\n'
        else:
            raise NotImplementedError('Cannot add newline to non-text placeholder')

    def __repr__(self):
        return ('DefaultSourceMatcher "{}" for node "{}" expecting to match "{}"'
                .format(super(DefaultSourceMatcher, self).__repr__(),
                        self.node,
                        pprint.pformat(self.expected_parts)))
