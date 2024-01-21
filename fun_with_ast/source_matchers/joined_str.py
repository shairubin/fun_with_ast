import ast
import re
from string import Formatter

from fun_with_ast.get_source import GetSource
from fun_with_ast.placeholders.list_placeholder import ListFieldPlaceholder
from fun_with_ast.placeholders.text import TextPlaceholder
from fun_with_ast.source_matchers.defualt_matcher import DefaultSourceMatcher
from fun_with_ast.source_matchers.joined_str_config import JstrConfig, SUPPORTED_QUOTES


class JoinedStrSourceMatcher(DefaultSourceMatcher):
    MAX_LINES_IN_JSTR = 10
    NEW_IMPLEMENTATION = False
    def __init__(self, node, starting_parens=None, parent=None):
        raise NotImplementedError('Depricated - use JoinedStrSourceMatcherNew')
        expected_parts = [
            TextPlaceholder(r'f[\'\"]', 'f\''),
            ListFieldPlaceholder(r'values'),
            TextPlaceholder(r'[\'\"][ \t\n]*', '', no_transform=True),
        ]
        super(JoinedStrSourceMatcher, self).__init__(
            node, expected_parts, starting_parens)
        self.padding_quote = None
        self.jstr_meta_data: list[JstrConfig] = []



    def _match(self, string):
        raise NotImplementedError('Depricated - use JoinedStrSourceMatcherNew')
        self.orig_string = string
        remaining_string = self.MatchStartParens(string)
        self._split_jstr_into_lines(remaining_string)
        self.padding_quote = self.jstr_meta_data[0].quote_type
        multi_part_string = self._convert_to_multi_part_string()
        embedded_string = multi_part_string
        # default string matcher will match the multipart string
        if not self.NEW_IMPLEMENTATION:
            matched_text = super(JoinedStrSourceMatcher, self)._match(embedded_string)
            self.matched = False # ugly hack to force the next line to work
            self.matched_source = None
            len_jstr = self._get_size_of_jstr_string()
            remaining_string = remaining_string[len_jstr:]
            remaining_string = self.MatchEndParen(remaining_string)
            remaining_string = self.MatchCommentEOL(remaining_string)
        else:
            matched_text_new = self._match_new(string)
            self.matched = False # ugly hack to force the next line to work
            self.matched_source = None
            #len_jstr = self._get_size_of_jstr_string()
            remaining_string = remaining_string[len(matched_text_new):]
            remaining_string = self.MatchEndParen(remaining_string)
            remaining_string = self.MatchCommentEOL(remaining_string)

        matched_text = self.GetSource()
        self.matched_source = matched_text
        self.matched = True
        return matched_text

    def GetSource(self):
        if self.matched:
            return self.matched_source
        matched_source = super(JoinedStrSourceMatcher, self).GetSource()
        matched_source = self._convert_to_single_part_string(matched_source)
        matched_source = self._split_back_into_lines(matched_source)
        return matched_source

    def _convert_to_multi_part_string(self):
        multi_part_result = self.jstr_meta_data[0].f_part
        format_string = ''
        for config in self.jstr_meta_data:
            format_string += config.format_string
        if format_string == '':
            return multi_part_result + self.padding_quote*3
        format_parts = list(Formatter().parse(format_string))
        for (literal, name, format_spec, conversion) in format_parts:
            if literal:
                multi_part_result += self.padding_quote + literal + self.padding_quote
            if name:
                multi_part_result += self.padding_quote + '{' + name + '}' + self.padding_quote
            if format_spec:
                raise NotImplementedError("format_spec not supported in format string yet")
            if conversion:
                pass
        multi_part_result = self.jstr_meta_data[0].prefix_str + multi_part_result + self.padding_quote
        if self.jstr_meta_data[0].f_part_type != 'f':
            # a joined string that its first element is not 'f' e.g., "X"\nf"Y"
            multi_part_result = 'f' + multi_part_result
        return multi_part_result

    def _convert_to_single_part_string(self, _in):
        result = _in
        result = result.replace(self.padding_quote * 2, self.padding_quote)
        if result == self.orig_string:
            return result
        prefix, suffix = self._get_prefix_suffix()
        if not self.start_paren_matchers and not result.startswith(prefix + 'f'+self.padding_quote):
                raise ValueError('We must see f\' at beginning of match')
        format_string = result.removesuffix(suffix)
        if not format_string.endswith(self.padding_quote) and not self.end_paren_matchers:
                raise ValueError("We must see \' or ')' at the end of match")
        format_string = self._verify_format_string(prefix, result, suffix)
        end_result = ''
        for start_paren in self.start_paren_matchers:
            end_result += start_paren.matched_text
        end_result += (prefix + 'f'+self.padding_quote + format_string + self.padding_quote)
        for end_paren in self.end_paren_matchers:
            end_result += end_paren.matched_text
        return end_result

    def _verify_format_string(self, prefix, result, suffix):
        reconstructed_format_string = self.__reconstruct_format_from_matcher_result(prefix, result, suffix)
        original_format = ''
        conversion_exists = False
        for config in self.jstr_meta_data:
            original_format += config.format_string
            conversion_exists = conversion_exists or config.conversion
        if reconstructed_format_string != original_format:
            for end_paren in self.end_paren_matchers:
                reconstructed_format_string = reconstructed_format_string.removesuffix(end_paren.matched_text)
            if reconstructed_format_string != original_format:
                if not conversion_exists:
                    raise ValueError('format string does not match')
                self.__verify_format_string_with_conversion(original_format, reconstructed_format_string)
        return original_format

    def __reconstruct_format_from_matcher_result(self, prefix, result, suffix):
        reconstructed_format_string = result
        for start_paren in self.start_paren_matchers:
            reconstructed_format_string = reconstructed_format_string.removeprefix(start_paren.matched_text)
        reconstructed_format_string = reconstructed_format_string.removeprefix(prefix + 'f' + self.padding_quote)
        reconstructed_format_string = reconstructed_format_string.removesuffix(suffix)
        for end_paren in self.end_paren_matchers:
            reconstructed_format_string = reconstructed_format_string.removesuffix(end_paren.matched_text)
        reconstructed_format_string = reconstructed_format_string.removesuffix(self.padding_quote)
        reconstructed_format_string = reconstructed_format_string.replace(self.padding_quote + '{', '{')
        reconstructed_format_string = reconstructed_format_string.replace('}' + self.padding_quote, '}')
        return reconstructed_format_string

    def _get_prefix_suffix(self):
        prefix_before_f = self.jstr_meta_data[0].prefix_str
        suffix_after_jsdr = self.jstr_meta_data[len(self.jstr_meta_data) - 1].suffix_str
        return prefix_before_f, suffix_after_jsdr



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


    def _embed_multiline_jstr_into_string(self, single_line_jstr, string):
        multiline_parts = self.jstr_meta_data.multiline_parts
        first_part = multiline_parts[0]["line"]
        last_part = multiline_parts[len(multiline_parts)-1]["line"]
        first_part_loc = string.find(first_part)
        if first_part_loc == -1:
            raise ValueError('could not identify first part')
        last_part_loc = string.find(last_part)
        if last_part_loc == -1:
            raise ValueError('could not identify last part')
        prefix = string[:first_part_loc]
        suffix = string[last_part_loc+len(last_part):]
        result = prefix + single_line_jstr + suffix
        return result

    def _split_matched_string_into_multiline(self, matched_text):
        for config in self.jstr_meta_data:
            format_string = matched_text.find(config.format_string)
            if format_string == -1:
                raise NotImplementedError

    def _is_jstr(self, line, line_index):
        if line_index > 0 and isinstance(self.node.parent_node, (ast.Dict, ast.List, ast.Tuple)):
            return False # we assume that the dict /listhas only one-liners as jstr
        for quote in SUPPORTED_QUOTES:
            expr = r'^[ \t]*f?' + quote
            match = re.match(expr, line)
            if match:
                return True
        return False

    def _split_back_into_lines(self, matched_text):
        result=''
        if len(self.jstr_meta_data) == 1:
            return matched_text
        remaining_string = matched_text
        for index, config in enumerate(self.jstr_meta_data):
            if index == 0:
                for start_paren in self.start_paren_matchers:
                    result += start_paren.matched_text
                config_contrib = config.prefix_str + config.f_part+config.format_string + config.quote_type
            else:
                config_contrib = config.prefix_str + config.f_part+  config.format_string + config.quote_type
            result += config_contrib
            if index != len(self.jstr_meta_data)-1:
                result += '\n'
        for end_paren in self.end_paren_matchers:
            result += end_paren.matched_text
        return result


    def _get_size_of_jstr_string(self):
        result = 0
        for config in self.jstr_meta_data:
            result += config.jstr_length
        return result + len(self.jstr_meta_data) - 1

    def __verify_format_string_with_conversion(self, original_format, reconstructed_format_string):
        if len(original_format) <= len(reconstructed_format_string):
            raise ValueError('format string does not contain conversion')
        reconstruct_index = 0
        original_index = 0
        while original_index < len(original_format):
            orig = original_format[original_index]
            recon = reconstructed_format_string[reconstruct_index]
            if orig == recon:
                reconstruct_index += 1
                original_index += 1
                continue
            conversion = original_format[original_index:original_index+2]
            if conversion in  ['!r', '!a', '!s']:
                original_index += 2
            else:
                raise ValueError('format string does not match reconstruction')

    def _match_new(self, orig_string):
        matched_text = super(JoinedStrSourceMatcher, self)._match(orig_string)
        result = ''
        remaining_string = orig_string
        remaining_string = remaining_string.removeprefix('f')
        node = self.node
        if len(node.values) > 1:
            raise NotImplementedError('not implemented yet')
        for value in node.values:
            if isinstance(value, ast.Constant):
                source = GetSource(value,  parent_node=node)
                result += source
                remaining_string = remaining_string[len(result):]
            else:
                raise NotImplementedError('not implemented yet')
        return 'f'+ result

def GetElements(self, node):
    raise NotImplementedError('not implemented yet')