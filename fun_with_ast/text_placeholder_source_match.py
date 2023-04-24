# import re
#
# from placeholder_source_match import Placeholder
# from exceptions_source_match import BadlySpecifiedTemplateError
#
#
import re


from fun_with_ast.exceptions_source_match import BadlySpecifiedTemplateError
from fun_with_ast.placeholder_source_match  import Placeholder


class TextPlaceholder(Placeholder):
    """Placeholder for text (non-field). For example, 'def (' in FunctionDef."""

    def __init__(self, regex, default=None):
        super(TextPlaceholder, self).__init__()
        self.original_regex = regex
#        self.regex = regex
        self.regex = self._TransformRegex(regex)
        if default is None:
            self.default = regex
        else:
            self.default = default
        self.matched_text = None

    def _TransformRegex(self, regex):
        non_whitespace_parts = regex.split(r'\s*')
        regex = r'\s*(\\\s*|#.*\s*)*'.join(non_whitespace_parts)
#        regex = r'\s*(\\\s*)*'.join(non_whitespace_parts)
        non_linebreak_parts = regex.split(r'\n')
        regex = r'( *#.*\n| *;| *\n)'.join(non_linebreak_parts)
#        regex = r'( *;| *\n)'.join(non_linebreak_parts)
        return regex

    def Match(self, unused_node, string, dotall=False):
        """Attempts to match string against self.regex.

        Saves the matched section for use in GetSource.

        Args:
          unused_node: unused.
          string: The string we attempt to match a substring of.
          dotall: Whether to apply re.DOTALL to the match.

        Raises:
          BadlySpecifiedTemplateError: If the regex doesn't match anywhere.

        Returns:
          The substring of string that matches.
        """
        if dotall:
            match_attempt = re.match(self.regex, string, re.DOTALL)
        else:
            match_attempt = re.match(self.regex, string)
            #all = re.findall(self.regex, string)
        if not match_attempt:
            raise BadlySpecifiedTemplateError(
                'string "{}" does not match regex "{}" (technically, "{}")'
                    .format(string, self.original_regex, self.regex))
        self.matched_text = match_attempt.group(0)
        return self.matched_text

    def GetSource(self, unused_node):
        """Returns self.matched_text if it exists, or self.default otherwise."""
        if self.matched_text is None:
            return self.default
        return self.matched_text

    def Copy(self):
        return TextPlaceholder(self.regex, self.default)

    def __repr__(self):
        return 'TextPlaceholder with regex "{}" ("{}") and default "{}"'.format(
            self.original_regex, self.regex, self.default)
