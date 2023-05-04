from string import Formatter

from fun_with_ast.source_matchers.exceptions_source_match import BadlySpecifiedTemplateError
from fun_with_ast.defualt_source_matcher_source_match import DefaultSourceMatcher
from fun_with_ast.list_placeholder_source_match import ListFieldPlaceholder
#from fun_with_ast.source_matchers.str_source_match import StrSourceMatcher
#from fun_with_ast.string_part_placeholder import JoinedStringPartPlaceholder
#from fun_with_ast.source_matcher_source_match import MatchPlaceholder
from fun_with_ast.text_placeholder_source_match import TextPlaceholder


class JoinedStrSourceMatcher(DefaultSourceMatcher):
    """Source matcher for _ast.Tuple nodes."""

    def __init__(self, node, starting_parens=None, parent=None):
        expected_parts = [
            TextPlaceholder(r'f[\'\"]', 'f\''),
            ListFieldPlaceholder(r'values'),
            TextPlaceholder(r'[\'\"]', '')
        ]
        super(JoinedStrSourceMatcher, self).__init__(
            node, expected_parts, starting_parens)
        self.padding_quote = None


    def Match(self, string):
        self.padding_quote = self._get_padding_quqte(string)
        string = self._convert_to_multi_part_string(string)
        matched_text = super(JoinedStrSourceMatcher, self).Match(string)
        return matched_text

    def _convert_to_multi_part_string(self, _in):
        formatted_string = list(Formatter().parse(_in[2:]))
#        if len(formatted_string) == 1 and formatted_string[0][1] == None:
#            return _in
        multi_part = _in[0:2]
        for (literal, name, format_spec, conversion) in formatted_string:
            print(literal, name, format_spec, conversion)
            if literal:
                multi_part += self.padding_quote + literal + self.padding_quote
            if name:
                multi_part += '\'{' + name + '}\''
            if format_spec:
                raise NotImplementedError
            if conversion:
                raise NotImplementedError

        return multi_part

    def MatchStartParens(self, remaining_string):
        return remaining_string

    def GetSource(self):
        matched_source = super(JoinedStrSourceMatcher, self).GetSource()
        matched_source = self._convert_to_single_part_string(matched_source)

        return matched_source

    def _convert_to_single_part_string(self, _in):
        if _in[-2:] == self.padding_quote * 2:
            result = _in[:-1]
        if result[0:3] == "f" + self.padding_quote*2:
            result = result.replace("f"+self.padding_quote*2, "f"+ self.padding_quote)
        return result

    def _get_padding_quqte(self, string):
        if string.startswith("f'"):
            return "'"
        elif string.startswith("f\""):
            return "\""
        raise BadlySpecifiedTemplateError('Formatted string must start with \' or \"')
# class JoinedStrSourceMatcher(StrSourceMatcher):
#     def __init__(self, node, starting_parens=None):
#         super(JoinedStrSourceMatcher, self).__init__(node, starting_parens)
#         self.value_placeholder = ListFieldPlaceholder('values')
#
#     def _get_original_string(self):
#         self.original_s = ''
#
#     def _part_place_holder(self):
#         return JoinedStringPartPlaceholder()
#
#     def Match(self, string):
#         part = self._part_place_holder()
#         remaining_string = MatchPlaceholder(string, None, part)
#         self.quote_parts.append(part)
#
#     def _handle_multipart(self, remaining_string):
#         pass