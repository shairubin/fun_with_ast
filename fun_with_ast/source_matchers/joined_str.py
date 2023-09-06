import re
from dataclasses import dataclass, field
from string import Formatter

from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.defualt_matcher import DefaultSourceMatcher
from fun_with_ast.placeholders.list_placeholder import ListFieldPlaceholder
from fun_with_ast.placeholders.text import TextPlaceholder

supported_quotes = ['\'', "\""]
@dataclass
class JstrConfig:
    orig_single_line_string: str
    prefix_re: re.Match
    suffix_re: re.Match
    prefix_str: str
    suffix_str: str
    f_part: str
    f_part_location: int
    format_string: str
    end_quote_location: int
    matched_multipart_string: str
    quote_type: str

    def __init__(self, line):
        self.orig_single_line_string = line
        self._create_config()

    def _create_config(self):
        self.suffix_str = ''
        self.prefix_str = ''
        self._set_quote_type()
        self._set_f_prefix()
        self.end_quote_location = self.orig_single_line_string.rfind(self.quote_type)
        if self.end_quote_location == -1:
            raise ValueError('Could not find ending quote')
        self.suffix_str= self.orig_single_line_string[self.end_quote_location+1:]
        self.prefix_str = self.orig_single_line_string[:self.f_part_location]
#        prefix_pattern = '([ \t\(])*(f'+self.quote_type+')'
#        self.prefix_re = re.match(prefix_pattern, self.orig_single_line_string)
#        if self.prefix_re:
#            self.prefix_str = '' if not self.prefix_re.group(1) else self.prefix_re.group(1)
#            self.f_part = self.prefix_re.group(2)
#        else:
#            raise ValueError('We must have f_part')
        # suffix_pattern = "(.*(\")([ \t]*\)?))$"
        #                   #""
        #                   #" "(\\"+self.quote_type +   "[ \t]*\)?)$"
        #suffix_pattern = "(.*([ \t]*\)?))$"
        #self.suffix_re = re.match(suffix_pattern, self.orig_single_line_string)
        #if self.suffix_re:
        #    self.suffix_str = self.suffix_re.group(1)
        self.format_string = self.orig_single_line_string
        #if self.suffix_str:
        self.format_string = self.format_string.removesuffix(self.quote_type + self.suffix_str)
        self.format_string = self.format_string.removeprefix(self.prefix_str+self.f_part)
        #if self.format_string[-1] != '\'':
        #    ValueError('at this point format_string must end with quate')
        #self.format_string = self.format_string[:-1]



    def _set_quote_type(self):
        for quote in supported_quotes:
            if self.orig_single_line_string.find("f"+quote) != -1:
                self.quote_type = quote
                return
        raise ValueError("could not find quote in singlke line string")

    def _set_f_prefix(self):
        location = self.orig_single_line_string.find("f"+self.quote_type)
        if location == -1:
            raise ValueError("could not find quote in singlke line string")
        self.f_part = self.orig_single_line_string[location:location+2]
        self.f_part_location = location
class JoinedStrSourceMatcher(DefaultSourceMatcher):
    """Source matcher for _ast.Tuple nodes."""
    USE_NEW_IMPLEMENTATION = True

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
        self.orig_string = string
        self._split_jstr_into_lines(string)
        self.padding_quote = self.jstr_meta_data[0].quote_type
        #jstr = self._extract_jstr_string(string, False)
        jstr = self._generate_to_multi_part_string()
        if self.USE_NEW_IMPLEMENTATION:
            embeded_string = self._embed_jstr_into_string(jstr, string)
            matched_text = super(JoinedStrSourceMatcher, self)._match(embeded_string)
        else:
            raise NotImplementedError('deprecated')
        return matched_text

    def _generate_to_multi_part_string(self,):
        if not self.USE_NEW_IMPLEMENTATION:
            raise NotImplementedError('deprecated')
        else:
            # if not _in.startswith("f"):
            #     raise ValueError("formatted string must start with f")
            # if _in[1] != self.padding_quote:
            #     raise ValueError("_in[1] must be a padding quote")
            # if not _in.endswith(self.padding_quote):
            #     raise ValueError("formatted string must end with '")
            # format_string = _in[2:-1]
            multi_part_result = self.jstr_meta_data[0].f_part
            format_string = ''
            for config in self.jstr_meta_data:
                format_string += config.format_string
            if format_string == '':
                return multi_part_result + self.padding_quote
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
            multi_part_result += self.padding_quote
        return multi_part_result

    def GetSource(self):
        matched_source = super(JoinedStrSourceMatcher, self).GetSource()
        matched_source = self._convert_to_single_part_string(matched_source)
        matched_source = self._split_back_into_lines(matched_source)

        return matched_source

    def _convert_to_single_part_string(self, _in):
        if not self.USE_NEW_IMPLEMENTATION:
            raise NotImplementedError('deprecated')
        else: # TODO kind of ugly here
            #extracted_multipart_string = self._extract_jstr_string(_in, True)
#            extracted_multipart_string = self.jstr_meta_data.matched_multipart_string
#            result = extracted_multipart_string
            result = _in
            result = result.replace(self.padding_quote * 2, self.padding_quote)
            if result == self.orig_string:
                return result
            prefix = self.jstr_meta_data[0].prefix_str
            suffix = self.jstr_meta_data[len(self.jstr_meta_data)-1].suffix_str
            if not result.startswith(prefix + 'f'+self.padding_quote):
                raise ValueError('We must see f\' at beginning of match')
            if not result.endswith(self.padding_quote+suffix):
                raise ValueError('We must see \' at the end of match')

            tmp_format_string = result.removeprefix(prefix + 'f'+self.padding_quote)
            tmp_format_string = tmp_format_string.removesuffix( self.padding_quote + suffix )

            #result = result.replace("f"+self.padding_quote*2, "f"+ self.padding_quote)
            tmp_format_string=tmp_format_string.replace(self.padding_quote+'{', '{')
            tmp_format_string =tmp_format_string.replace('}'+self.padding_quote, '}')
            #if not result.endswith(self.padding_quote):
            #    if self.jstr_meta_data.format_string not in result:
            #        result += self.padding_quote
            #result = self._embed_jstr_into_string(result, _in, True)
            result = prefix + 'f'+self.padding_quote +tmp_format_string + self.padding_quote + suffix
            return result

    def _get_padding_quqte(self, string):
        if string.startswith("f'"):
            return "'"
        elif string.startswith("f\""):
            return "\""
        raise BadlySpecifiedTemplateError('Formatted string must start with \' or \"')

#    def _check_not_implemented(self, string):
#        if '\"\"' in string:
#            raise NotImplementedError('Double-quotes are not supported yet')



    # def _extract_jstr_string(self, string ,is_double_quaoted):
    #     if not is_double_quaoted:
    #         raise NotImplementedError
    #     f_part = = self.jstr_meta_data[0].prefix_str
    #
    #     suffix = self.jstr_meta_data[len(self.jstr_meta_data)-1].suffix_str
    #     result = string.split(suffix)[0]
    #     result = result.split(prefix)[1]
    #     return result
        # result = self.jstr_meta_data[0].f_part + result
        # return result

        # end, start = self._find_start_end_of_jstr(string)
        # extracted_string = string[start:end+1]
        # stripped_string = string.strip()
        # if stripped_string != extracted_string:
        #     if stripped_string[end+1] not in [')', '\n']:
        #         raise NotImplementedError("extracted_string is not followed by ')'")
        # self._save_meta_data(end, extracted_string, is_multi_part, start)

    # def _find_start_end_of_jstr(self, string):
    #
    #     start = string.find("f'")
    #     if start == -1:
    #         start = string.find("f\"")
    #         if start == -1:
    #             raise BadlySpecifiedTemplateError('Formatted string must start with \' or \"')
    #         end = self._guess_end_of_jstr(string, "\"")
    #         if end == -1:
    #             raise BadlySpecifiedTemplateError('Formatted string must end with \"')
    #     else:
    #         end = self._guess_end_of_jstr(string, "'")
    #         if end == -1:
    #             raise BadlySpecifiedTemplateError('Formatted string must end with \'')
    #     return end, start

    def _guess_end_of_jstr(self, string, quote):
        lines = string.split('\n')
        #lines = self.lines
        if len(lines) > 1 and lines[1].strip().startswith('f'):
            if self.jstr_meta_data.multiline_jstr == True:
                raise ValueError('When multiline_jstr set to True, single line is allowed only')
            raise MultiPartJoinedString
        else:
            jstr_end = lines[0].rfind(quote)

        new_line = string.find('\n')
        if new_line == -1 :
            end = string.rfind(quote)
        else:
            end = string.rfind(quote, 0, new_line)
        if jstr_end != end:
            raise NotImplementedError('jstr_end != end')
        return end

    def _save_meta_data(self, end, extracted_string, is_multi_part, start):
        if not is_multi_part:
            self.jstr_meta_data.format_string = extracted_string
            self.jstr_meta_data.format_start_at = start
            self.jstr_meta_data.format_end_at = end
        else:
            self.jstr_meta_data.matched_multipart_string = extracted_string
            self.jstr_meta_data.multipart_start_at = start
            self.jstr_meta_data.multipart_end_at = end

    def _embed_jstr_into_string(self, jstr, string):
        # if self.jstr_meta_data.multiline_jstr:
        #     raise NotImplementedError
        #     return self._embed_multiline_jstr_into_string(jstr, string)
        # if not is_multi_part:
        #     jstr_start = self.jstr_meta_data.format_start_at
        #     jstr_end = self.jstr_meta_data.format_end_at
        # else:
        #     jstr_start = self.jstr_meta_data.multipart_start_at
        #     jstr_end = self.jstr_meta_data.multipart_end_at
        # prefix = string[:jstr_start]
        # suffix = string[jstr_end+1:]
        # result = prefix + jstr + suffix
        # return result

#        if len(self.jstr_meta_data) > 1:
#            raise NotImplementedError
        prefix = self.jstr_meta_data[0].prefix_str
        suffix = self.jstr_meta_data[len(self.jstr_meta_data)-1].suffix_str
        result = prefix + jstr + suffix
        return result

    def _split_jstr_into_lines(self, orig_string):
        lines = orig_string.split('\n')
        jstr_lines = []
        for line in lines:
            if line.find("f'") != -1 or line.find("f") != -1:
                jstr_lines.append(line)
            else:
                break
#        if len(jstr_lines) > 1 :
#            raise NotImplementedError
        for line in jstr_lines:
            if self._is_jstr(line):
                self.jstr_meta_data.append(JstrConfig(line))
            else:
                raise ValueError

    # def _extract_multiline_jstr(self, string):
    #     lines = string.split('\n')
    #     single_jstr = ''
    #     self.jstr_meta_data.multiline_jstr = True
    #     for line in  lines:
    #         if line.find("f'") != -1 or line.find("f\"") != -1:
    #             end, start = self._find_start_end_of_jstr(line)
    #             self.jstr_meta_data.multiline_parts.append({"line": line, "start": start, "end": end})
    #             single_jstr += line[start+2:end]
    #         else:
    #             break
    #     single_jstr = "f'" + single_jstr + "'"
    #     self._extract_jstr_string(single_jstr, False)

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

    def _is_jstr(self, line):
        for quote in supported_quotes:
            expr = f'[ \t\(]*f'+quote
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
                config_contrib = config.prefix_str + config.f_part+config.format_string
            else:
                config_contrib = config.format_string
            line_start_at = remaining_string.find(config_contrib)
            if line_start_at == -1:
                ValueError('invalid match of line in multiline jstr string')
            if line_start_at != 0:
                ValueError('single line must be the start of the multiline jstr string')
            result += config.orig_single_line_string
            if index != len(self.jstr_meta_data)-1:
                result += '\n'
            remaining_string = remaining_string.removeprefix(config_contrib)
        return result

