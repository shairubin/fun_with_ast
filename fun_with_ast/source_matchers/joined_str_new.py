import ast
import re
from string import Formatter

from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.placeholders.base_match import MatchPlaceholder
from fun_with_ast.placeholders.node import NodePlaceholder
from fun_with_ast.manipulate_node.nodes_for_jstr import ConstantForJstr, NameForJstr
from fun_with_ast.source_matchers.joined_str_config import SUPPORTED_QUOTES, JstrConfig

from fun_with_ast.source_matchers.defualt_matcher import DefaultSourceMatcher
from fun_with_ast.placeholders.text import TextPlaceholder
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class JoinedStrSourceMatcherNew(DefaultSourceMatcher):
    MAX_LINES_IN_JSTR = 10
    def __init__(self, node, starting_parens=None, parent=None):
        expected_parts = [
         TextPlaceholder(r'f[\'\"]', 'f\''),
         TextPlaceholder(r'[\'\"]', '\'')
     ]
        super(JoinedStrSourceMatcherNew, self).__init__(
            node, expected_parts, starting_parens)
        self.padding_quote = None
        self.jstr_meta_data = []
        #self.end_whitespace_matchers = [TextPlaceholder(r'[ \t\n]*', '')]


    def _match(self, string):
        remaining_string = self.MatchStartParens(string)
        self._split_jstr_into_lines(remaining_string)
        source = ''

        if len(self.jstr_meta_data)>1: # this is multi line jstr
            self._mark_node_values_as_potentially_matched()
            for index, line in enumerate(self.jstr_meta_data):
                if line.prefix_str:
                    #raise NotImplementedError('prefix string not supported yet')
                    string_to_match = line.full_jstr_including_prefix.removeprefix(line.prefix_str)
                else:
                    string_to_match = line.full_jstr_including_prefix.removeprefix(line.prefix_str)
                one_line_node =GetNodeFromInput(string_to_match)
                matcher = GetDynamicMatcher(one_line_node)
                matcher.end_whitespace_matchers = [TextPlaceholder(r'[ \t]*', '', no_transform=True)]
                matched_line_text = line.prefix_str +  matcher._match(string_to_match)
                #matched_line_text = line.prefix_str +  matcher._match(line.full_jstr_including_prefix)
                if index != len(self.jstr_meta_data)-1:
                    matcher.matched_source = matched_line_text +"\n"
                    source += matched_line_text +"\n"
                else:
                    matcher.matched_source = matched_line_text
                    source += matched_line_text
                self.expected_parts.insert(index + 1, NodePlaceholder(one_line_node))
            self.expected_parts.pop(0)
            self.expected_parts.pop()
            remaining_string = remaining_string[len(source):]
            self._conclude_jstr_match(remaining_string)

        elif len(self.jstr_meta_data) == 1:
            source =  self._match_single_line_jstr(remaining_string,0)
        return source

    def _match_single_line_jstr(self, remaining_string, index):
        remaining_string = MatchPlaceholder(remaining_string, self.node, self.expected_parts[0])
        format_string = self._get_format_string(index)
        format_parts = self._get_format_parts(format_string)
        format_string_source = self._match_format_parts(format_parts)
        # if format_string_source != format_string:
        #    raise ValueError('format strings does not match')
        # source =   self.jstr_meta_data[0].f_part + format_string_source
        self.matched = False  # ugly hack to force the next line to work
        self.matched_source = None
        # len_jstr = self._get_size_of_jstr_string()
        remaining_string = remaining_string[len(format_string_source):]
        remaining_string = MatchPlaceholder(remaining_string, self.node, self.expected_parts[-1])
        return self._conclude_jstr_match( remaining_string)

    def _conclude_jstr_match(self, remaining_string):
        remaining_string = self.MatchWhiteSpaces(remaining_string, self.end_whitespace_matchers[0])
        remaining_string = self.MatchEndParen(remaining_string)
        remaining_string = self.MatchCommentEOL(remaining_string)
        matched_source = self.GetSource()
        self.matched = True
        self.matched_source = matched_source
        return matched_source

    def _get_format_string(self, index):
        format_string = ''
        config = self.jstr_meta_data[index]
        format_string = config.format_string
#         for config in self.jstr_meta_data:
#             if format_string_to_match is None or format_string_to_match == '':
#                 format_string += config.format_string
#             else:
#                 raise ValueError('deprecated')
#                 bare_format_string = config.format_string.removeprefix('{').removesuffix('}')
# #                if bare_format_string.strip() == format_string_to_match:
#                 if bare_format_string == format_string_to_match:
#                     format_string =  config.format_string
#                     break
#        if  format_string_to_match and format_string == '':
#            raise ValueError('format string not found')
        return format_string

    def _get_format_parts(self, format_string):
        format_parts = list(Formatter().parse(format_string))
        return format_parts
    def _get_quote_type(self):
        return self.jstr_meta_data[0].quote_type







#     def _use_default_matcher(self, string):
# #        parts = self._get_parts_for_default_matcher(self.node)
#         self.values_matcher = GetDynamicMatcher(self.node, parts_in=parts)
#         self.values_matcher.EOL_matcher = None # args will not end with '\n' -- parent node will consume it
#         matched_string = self.values_matcher._match(string)
#         return matched_string

    # def _get_parts_for_default_matcher(self, node):
    #     expected_parts =[]
    #     expected_parts.append(TextPlaceholder(r'f', 'f'))
    #     if len(self.node.values) != 1:
    #         raise NotImplementedError('Only one value is supported')
    #     expected_parts.append(ListFieldPlaceholder(r'values'))
    #     expected_parts.append(TextPlaceholder(r'[\'\"][ \t\n]*', '', no_transform=True))
    #     return expected_parts

    def _split_jstr_into_lines(self, orig_string):
        if isinstance(self.node.parent_node, ast.Dict):
            lines = re.split(r'[\n:]', orig_string, maxsplit=self.MAX_LINES_IN_JSTR*2)
        elif isinstance(self.node.parent_node, (ast.List, ast.Tuple)):
            lines = re.split(r'[\n,]', orig_string, maxsplit=self.MAX_LINES_IN_JSTR*2)
        else:
            lines = orig_string.split('\n', self.MAX_LINES_IN_JSTR)
        jstr_lines = []
        for index, line in enumerate(lines):
            if self._is_jstr(line, index):
                jstr_lines.append(line)
            else:
                break
        if len(jstr_lines) >= self.MAX_LINES_IN_JSTR-1:
            raise ValueError('too many lines in jstr string')
        self._update_jstr_meta_data_based_on_context(jstr_lines, lines)
    # not clear why we need this fucntion below
    def _update_jstr_meta_data_based_on_context(self, jstr_lines, lines): # this function mostly for debugging purposes
        if len(jstr_lines) == 0:
            raise ValueError('could not find jstr lines')
        if len(jstr_lines) == 1: # simple case
            self.__appnd_jstr_lines_to_metadata(jstr_lines)
            return
        last_jstr_line = jstr_lines[len(jstr_lines)-1]
        len_jstr_lines = len(jstr_lines)
        if re.search(r'[ \t]*\)[ \t]*$', last_jstr_line):      # this is call_args context
            self.__appnd_jstr_lines_to_metadata(jstr_lines)
            return
        elif len_jstr_lines < len(lines)  and re.match(r'[ \t\n]*\)', lines[len_jstr_lines]):
            self.__appnd_jstr_lines_to_metadata(jstr_lines) # this is call_args context
            return
        elif re.search(r'[ \t]*\)[ \t]*#.*$', last_jstr_line):  # this is call_args context
            self.__appnd_jstr_lines_to_metadata(jstr_lines)
            return
        elif re.search(r'[ \t]*\)[ \t]*from.*$', last_jstr_line):  # this is raise context
            self.__appnd_jstr_lines_to_metadata(jstr_lines)

        else:
            raise ValueError("Not supported - jstr string not in call_args context ")

    def __appnd_jstr_lines_to_metadata(self, jstr_lines):
        for line_index, line in enumerate(jstr_lines):
            self.jstr_meta_data.append(JstrConfig(line, line_index))

    def _is_jstr(self, line, line_index):
        if line_index > 0 and isinstance(self.node.parent_node, (ast.Dict, ast.List, ast.Tuple)):
            return False # we assume that the dict /listhas only one-liners as jstr
        for quote in SUPPORTED_QUOTES:
            expr = r'^[ \t]*f?' + quote
            match = re.match(expr, line)
            if match:
                return True
        return False

    def _match_format_parts(self, format_parts):
         format_string = ''
         #if len(format_parts) > 1:
         #       raise NotImplementedError('Only one format part is supported')
         index_in_jstr_values = 0
         for part in format_parts:
             literal_part = part[0]
             conversion_part = part[3]
             field_name_part = part[1:3]
             #if field_name_part[0] is not None:
             #    if field_name_part[1] is not '' or literal_part:
             #        raise NotImplementedError('named index not supported yet')
             format_string_literal = format_string_field_name = ''
             if literal_part:
                format_string_literal = self._handle_jstr_constant(index_in_jstr_values)
                index_in_jstr_values += 1
             if field_name_part[0] is not None:
                format_string_field_name = self._handle_jstr_field_name(index_in_jstr_values,
                                                                        conversion_part, field_name_part)
                index_in_jstr_values += 1
             format_string += format_string_literal + format_string_field_name
         return format_string
    def _handle_jstr_field_name(self, index, conversion, field_name_part):
        format_value_node = self.node.values[index]
        if not isinstance(format_value_node, ast.FormattedValue):
            raise ValueError('value node is not FormattedValue')
        value_node = format_value_node.value
        if not isinstance(value_node, (ast.Name, ast.Constant)):
            raise NotImplementedError('only name nodes are supported')

#        stripped_format  = GetSource(value_node, text=name_part[0])
        stripped_format  = GetSource(value_node)
        ws_parts = field_name_part[0].split(stripped_format)
        if len(ws_parts) != 2:
            raise ValueError('none whitespace in format string')
        stripped_format = ws_parts[0] + stripped_format + ws_parts[1]
        if stripped_format != field_name_part[0]:
            raise ValueError('format string does not match')
        if conversion:
            stripped_format = stripped_format + "!"+ conversion
        #name_node_for_jstr = NameForJstr(name_node)
        matcher = GetDynamicMatcher(format_value_node)
        format_string = "{" + stripped_format + "}"
        matched_string = matcher._match(format_string)

        #self.node.values[index] = constant_node_for_jstr
        self.expected_parts.insert(index + 1, NodePlaceholder(format_value_node))
        return format_string

    def _handle_jstr_constant(self, index):
        constant_node = self.node.values[index]
        value = constant_node.value
        constant_node_for_jstr = ConstantForJstr(value)
        constant_node_for_jstr.default_quote = self._get_quote_type()
        matched_string = GetSource(constant_node_for_jstr)
        self.node.values[index] = constant_node_for_jstr
        self.expected_parts.insert(index + 1, NodePlaceholder(constant_node_for_jstr))
        return matched_string

    def _mark_node_values_as_potentially_matched(self):
        for node in self.node.values:
            if isinstance(node, ast.FormattedValue):
                if isinstance(node.value, ast.Name):
                    node.value.no_matchers_ok = True
            node.no_matchers_ok = True



