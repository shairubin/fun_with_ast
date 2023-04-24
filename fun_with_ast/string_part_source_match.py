from fun_with_ast.placeholder_source_match import Placeholder
from fun_with_ast.string_parser import StringParser
from fun_with_ast.text_placeholder_source_match import TextPlaceholder
from fun_with_ast.utils_source_match import _FindQuoteEnd


class StringPartPlaceholder(Placeholder):
    """A container object for a single string part.

    Because of implicit concatination, a single _ast.Str node might have
    multiple parts.
    """

    def __init__(self):
        super(StringPartPlaceholder, self).__init__()
        self.prefix_placeholder = TextPlaceholder(r'ur|uR|Ur|UR|u|r|U|R|', '')
        self.quote_match_placeholder = TextPlaceholder(r'"""|\'\'\'|"|\'')
        self.inner_text_placeholder = TextPlaceholder(r'.*', '')

    def Match(self, node, string):
        elements = [self.prefix_placeholder, self.quote_match_placeholder]
        remaining_string = StringParser(string, elements).remaining_string

        quote_type = self.quote_match_placeholder.matched_text
        end_index = _FindQuoteEnd(remaining_string, quote_type)
        if end_index == -1:
            raise ValueError('String {} does not end properly'.format(string))
        self.inner_text_placeholder.Match(
            None, remaining_string[:end_index], dotall=True)
        remaining_string = remaining_string[end_index + len(quote_type):]
        if not remaining_string:
            return string
        return string[:-len(remaining_string)]

    def GetSource(self, node):
        placeholder_list = [self.prefix_placeholder,
                            self.quote_match_placeholder,
                            self.inner_text_placeholder,
                            self.quote_match_placeholder]
        source_list = [p.GetSource(node) for p in placeholder_list]
        return ''.join(source_list)
