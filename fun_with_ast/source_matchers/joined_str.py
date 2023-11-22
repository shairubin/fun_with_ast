import re
from dataclasses import dataclass, field
from string import Formatter

from fun_with_ast.source_matchers.defualt_matcher import DefaultSourceMatcher
from fun_with_ast.placeholders.list_placeholder import ListFieldPlaceholder
from fun_with_ast.placeholders.text import TextPlaceholder





supported_quotes = ['\'', "\""]
@dataclass
class JstrConfig:
    line_index: int
    orig_single_line_string: str
    prefix_str: str
    suffix_str: str
    f_part: str
    f_part_location: int
    format_string: str
    full_jstr_including_prefix: str
    end_quote_location: int
    start_quote_location: int
    quote_type: str
    jstr_length: int = 0
    f_part_type: str = 'not_set'
    def __init__(self, line, line_index):
        self.orig_single_line_string = line
        self._create_config(line_index)

    def _create_config(self, line_index):
        self.line_index = line_index
        self.suffix_str = ''
        self.prefix_str = ''
        self._set_quote_type()
        self._set_f_prefix()
        self.end_quote_location = self.orig_single_line_string.rfind(self.quote_type)
        self.start_quote_location = self.orig_single_line_string.find(self.quote_type)
        if self.start_quote_location == self.end_quote_location:
            raise ValueError('joined str string in which start and end quote locations are the same')
        if self.end_quote_location == -1:
            raise ValueError('Could not find ending quote')
        self.suffix_str = self.orig_single_line_string[self.end_quote_location+1:]
        #suffix_str_paren_only = re.match(r'[ \t]*\)[ \t]*', self.suffix_str)
        #if suffix_str_paren_only:
        #    self.suffix_str = suffix_str_paren_only.group(0)

        self.prefix_str = self.orig_single_line_string[:self.f_part_location]
        if self.prefix_str.strip() != '':
            raise ValueError('joined str string in which prefix is not white spaces')
        else:
            self.format_string = self.orig_single_line_string
        self.format_string = self.format_string.removesuffix(self.suffix_str)
        self.full_jstr_including_prefix = self.format_string
        self.format_string = self.format_string.removesuffix(self.quote_type)
        self.format_string = self.format_string.removeprefix(self.prefix_str+self.f_part)
        self.jstr_length = len(self.full_jstr_including_prefix)


    def _set_quote_type(self):
        for quote in supported_quotes:
#            if self.orig_single_line_string.find("f"+quote) != -1:
             if re.match(r'[ \t]*f?'+quote, self.orig_single_line_string):
                self.quote_type = quote
                return
        raise ValueError("could not find quote in single line string")

    def _set_f_prefix(self):
        (f_type, location) = self._set_prefix_type()
        if f_type == 'f':
            self.f_part = self.orig_single_line_string[location:location+2]
            self.f_part_location = location
            self.f_part_type = 'f'
        elif f_type == 'quote_only':
            self.f_part = self.orig_single_line_string[location:location+1]
            self.f_part_location = location
            self.f_part_type = 'quote_only'
        else:
            raise ValueError('could not find f or quote at the beginning of string')

    def _set_prefix_type(self):
        f_type = self.orig_single_line_string.find("f"+self.quote_type)
        if f_type != -1:
            return ('f',f_type)
        f_type = self.orig_single_line_string.find(self.quote_type)
        if f_type != -1 and self.line_index != 0:
            return ('quote_only', f_type)
        raise ValueError("could not find quote of f+quote in single line string")



class JoinedStrSourceMatcher(DefaultSourceMatcher):
    MAX_LINES_IN_JSTR = 10
    def __init__(self, node, starting_parens=None, parent=None):
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
        #remaining_string = string
        self.orig_string = string
        remaining_string = self.MatchStartParens(string)
        #self.orig_string = remaining_string
        self._split_jstr_into_lines(remaining_string)
        self.padding_quote = self.jstr_meta_data[0].quote_type
        multi_part_string = self._convert_to_multi_part_string()
        embedded_string = multi_part_string
        # default string match
        matched_text = super(JoinedStrSourceMatcher, self)._match(embedded_string)
        self.matched = False # ugly hack to force the next line to work
        self.matched_source = None
        #matched_text = self._convert_to_single_part_string(matched_text)
        #self._split_back_into_lines(matched_text)
        len_jstr = self._get_size_of_jstr_string()
        remaining_string = remaining_string[len_jstr:]
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

    def _convert_to_multi_part_string(self, ):
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
                raise NotImplementedError
            if conversion:
                raise NotImplementedError
        multi_part_result = self.jstr_meta_data[0].prefix_str + multi_part_result + self.padding_quote
        return multi_part_result

    def _convert_to_single_part_string(self, _in):
        result = _in
        result = result.replace(self.padding_quote * 2, self.padding_quote)
        if result == self.orig_string:
            return result
        prefix, suffix = self._get_prefix_suffix()
        if not self.start_paren_matchers:
            if not result.startswith(prefix + 'f'+self.padding_quote):
                raise ValueError('We must see f\' at beginning of match')
        format_string = result.removesuffix(suffix)
#        if not self.end_paren_matchers:
        if not format_string.endswith(self.padding_quote):
            if not self.end_paren_matchers:
                raise ValueError("We must see \' or ')' at the end of match")
#        else:
#            if not re.search(r'[ \t]*\)[ \t]*$', format_string):
#                raise ValueError('We must see ) at the end of match')
        format_string = self._verify_format_string(prefix, result, suffix)
        end_result = ''
        for start_paren in self.start_paren_matchers:
            end_result += start_paren.matched_text
        end_result += (prefix + 'f'+self.padding_quote + format_string + self.padding_quote)
        for end_paren in self.end_paren_matchers:
            end_result += end_paren.matched_text
        return end_result

    def _verify_format_string(self, prefix, result, suffix):
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
        original_format = ''
        for config in self.jstr_meta_data:
            original_format += config.format_string
        if reconstructed_format_string != original_format:
            for end_paren in self.end_paren_matchers:
                reconstructed_format_string = reconstructed_format_string.removesuffix(end_paren.matched_text)
            if reconstructed_format_string != original_format:
                raise ValueError('format string does not match')
        return reconstructed_format_string
    def _get_prefix_suffix(self):
        prefix_before_f = self.jstr_meta_data[0].prefix_str
        suffix_after_jsdr = self.jstr_meta_data[len(self.jstr_meta_data) - 1].suffix_str
        return prefix_before_f, suffix_after_jsdr



    def _split_jstr_into_lines(self, orig_string):
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

    def _update_jstr_meta_data_based_on_context(self, jstr_lines, lines):
        if len(jstr_lines) == 0:
            raise ValueError('could not find jstr lines')
        last_jstr_line = jstr_lines[len(jstr_lines)-1]
        first_jstr_line = jstr_lines[0]
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
        elif len(jstr_lines) == len(lines):                            # we assume this is module context
            self.jstr_meta_data.append(JstrConfig(first_jstr_line,0))
            return
        elif re.match(r'[ \t\n]*', lines[len_jstr_lines]):      # we assume this is module context
            self.jstr_meta_data.append(JstrConfig(first_jstr_line,0))
            return

        else:
            raise ValueError("unrecognized context for jst string")

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
        for quote in supported_quotes:
            if line_index == 0:
                expr = r'[ \t]*f'+quote
            else:
                expr = r'[ \t]*f?' + quote
            if re.search(expr,line):
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
            #self._get_start_line_location(config_contrib, remaining_string)
            result += config_contrib
            if index != len(self.jstr_meta_data)-1:
                result += '\n'
            #remaining_string = remaining_string.removeprefix(config_contrib)
        for end_paren in self.end_paren_matchers:
            result += end_paren.matched_text
        return result

    # def _get_start_line_location(self, config_contrib, remaining_string):
    #     line_start_at = remaining_string.find(config_contrib)
    #     if line_start_at == -1:
    #         ValueError('invalid match of line in multiline jstr string')
    #     if line_start_at != 0:
    #         ValueError('single line must be the start of the multiline jstr string')


    def _get_size_of_jstr_string(self):
        result = 0
        for config in self.jstr_meta_data:
            result += config.jstr_length
        return result + len(self.jstr_meta_data) - 1