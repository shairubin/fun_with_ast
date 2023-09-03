from dataclasses import dataclass, field
from string import Formatter

from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.defualt_matcher import DefaultSourceMatcher
from fun_with_ast.placeholders.list_placeholder import ListFieldPlaceholder
from fun_with_ast.placeholders.text import TextPlaceholder

@dataclass
class JstrMetaData:
    orig_string: str
    extracted_string: str
    start_at: int
    end_at: int
    matched_multipart_string: str
    multipart_start_at: int
    multipart_end_at: int
    multiline_jstr: bool = False
    multiline_parts: list = field(default_factory=list)


class MultiPartJoinedString(Exception):
    pass


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
        self.jstr_meta_data = JstrMetaData(None, None, None, None, None, None, None)
        self.lines = []



    def _match(self, string):
        self.jstr_meta_data.orig_string = string
        try:
            self._extract_jstr_string(string, False)
        except MultiPartJoinedString as e:
            self._extract_multiline_jstr(string)
        jstr= self.jstr_meta_data.extracted_string
        self._check_not_implemented(jstr)
        self.padding_quote = self._get_padding_quqte(jstr)
        jstr = self._convert_to_multi_part_string(jstr)
        if self.USE_NEW_IMPLEMENTATION:
            embeded_string = self._embed_jstr_into_string(jstr, string, False)
            matched_text = super(JoinedStrSourceMatcher, self)._match(embeded_string)
        else:
            raise NotImplementedError('deprecated')
        if self.jstr_meta_data.multiline_jstr:
            raise NotImplementedError('Not implemented yet')
        return matched_text

    def _convert_to_multi_part_string(self, _in):
        if not self.USE_NEW_IMPLEMENTATION:
            raise NotImplementedError('deprecated')
        else:
            if not _in.startswith("f"):
                raise ValueError("formatted string must start with f")
            if _in[1] != self.padding_quote:
                raise ValueError("_in[1] must be a padding quote")
            if not _in.endswith(self.padding_quote):
                raise ValueError("formatted string must end with '")
            format_string = _in[2:-1]
            format_parts = list(Formatter().parse(format_string))
            multi_part = _in[0:2]
            for (literal, name, format_spec, conversion) in format_parts:
                if literal:
                    multi_part += self.padding_quote + literal + self.padding_quote
                if name:
                    multi_part += self.padding_quote + '{' + name + '}' + self.padding_quote
                if format_spec:
                    raise NotImplementedError
                if conversion:
                    raise NotImplementedError
            multi_part += self.padding_quote
        return multi_part


    def GetSource(self):
        matched_source = super(JoinedStrSourceMatcher, self).GetSource()
        matched_source = self._convert_to_single_part_string(matched_source)

        return matched_source

    def _convert_to_single_part_string(self, _in):
        if not self.USE_NEW_IMPLEMENTATION:
            raise NotImplementedError('deprecated')
        else: # TODO kind of ugly here
            self._extract_jstr_string(_in, True)
            extracted_multipart_string = self.jstr_meta_data.matched_multipart_string
            result = extracted_multipart_string
            result = result.replace("f"+self.padding_quote*2, "f"+ self.padding_quote)
            result=result.replace(self.padding_quote*2+'{', '{')
            result=result.replace('}'+self.padding_quote*2, '}')
            if not result.endswith(self.padding_quote):
                if self.jstr_meta_data.extracted_string not in result:
                    result += self.padding_quote
            result=result.replace(self.padding_quote*2, self.padding_quote)
            result = self._embed_jstr_into_string(result, _in, True)
            return result

    def _get_padding_quqte(self, string):
        if string.startswith("f'"):
            return "'"
        elif string.startswith("f\""):
            return "\""
        raise BadlySpecifiedTemplateError('Formatted string must start with \' or \"')

    def _check_not_implemented(self, string):
        if '\"\"' in string:
            raise NotImplementedError('Double-quotes are not supported yet')



    def _extract_jstr_string(self, string ,is_multi_part):
        end, start = self._find_start_end_of_jstr(string)
        extracted_string = string[start:end+1]
        stripped_string = string.strip()
        if stripped_string != extracted_string:
            if stripped_string[end+1] not in [')', '\n']:
                raise NotImplementedError("extracted_string is not followed by ')'")
        self._save_meta_data(end, extracted_string, is_multi_part, start)

    def _find_start_end_of_jstr(self, string):

        start = string.find("f'")
        if start == -1:
            start = string.find("f\"")
            if start == -1:
                raise BadlySpecifiedTemplateError('Formatted string must start with \' or \"')
            end = self._guess_end_of_jstr(string, "\"")
            if end == -1:
                raise BadlySpecifiedTemplateError('Formatted string must end with \"')
        else:
            end = self._guess_end_of_jstr(string, "'")
            if end == -1:
                raise BadlySpecifiedTemplateError('Formatted string must end with \'')
        return end, start

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
            self.jstr_meta_data.extracted_string = extracted_string
            self.jstr_meta_data.start_at = start
            self.jstr_meta_data.end_at = end
        else:
            self.jstr_meta_data.matched_multipart_string = extracted_string
            self.jstr_meta_data.multipart_start_at = start
            self.jstr_meta_data.multipart_end_at = end

    def _embed_jstr_into_string(self, jstr, string, is_multi_part):
        if self.jstr_meta_data.multiline_jstr:
            self._embed_multiline_jstr_into_string(string)
        if not is_multi_part:
            jstr_start = self.jstr_meta_data.start_at
            jstr_end = self.jstr_meta_data.end_at
        else:
            jstr_start = self.jstr_meta_data.multipart_start_at
            jstr_end = self.jstr_meta_data.multipart_end_at
        prefix = string[:jstr_start]
        suffix = string[jstr_end+1:]
        result = prefix + jstr + suffix
        return result

    def _guess_split_of_jstr_into_multiline(self, lines):
        self.jstr_meta_data.multiline_jstr = True
        self.jstr_meta_data.multiline_parts = []
        for line in lines:
            if line.strip().startswith('f'):
                self.jstr_meta_data.multiline_parts.append(line)
            else:
                break

    def _extract_multiline_jstr(self, string):
        lines = string.split('\n')
        single_jstr = ''
        self.jstr_meta_data.multiline_jstr = True
        for line in  lines:
            if line.find("f'") != -1 or line.find("f\"") != -1:
                end, start = self._find_start_end_of_jstr(line)
                self.jstr_meta_data.multiline_parts.append({"line": line, "start": start, "end": end})
                single_jstr += line[start+2:end]
            else:
                break
        single_jstr = "f'" + single_jstr + "'"
        self._extract_jstr_string(single_jstr, False)

    def _embed_multiline_jstr_into_string(self, string):
        raise NotImplementedError
