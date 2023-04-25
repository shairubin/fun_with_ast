

import _ast
import pprint

import fun_with_ast.placeholder_source_match
from body_source_match import BodyPlaceholder
from fun_with_ast.args_placeholder_source_match import ArgsDefaultsPlaceholder, KeysValuesPlaceholder, ArgsKeywordsPlaceholder, \
    OpsComparatorsPlaceholder
from fun_with_ast.composite_placeholder_source_match import ListFieldPlaceholder, FieldPlaceholder

from fun_with_ast.exceptions_source_match import BadlySpecifiedTemplateError
from fun_with_ast.create_node import SyntaxFreeLine
from fun_with_ast.utils_source_match import _GetListDefault
from fun_with_ast.get_source import GetSource
from fun_with_ast.text_placeholder_source_match import TextPlaceholder, GetStartParenMatcher
from fun_with_ast.string_parser import StringParser



# TODO: Consolidate with StringParser
from fun_with_ast.source_matcher_source_match import SourceMatcher, MatchPlaceholder, MatchPlaceholderList




def StripStartParens(string):
    remaining_string = string
    while remaining_string.startswith('('):
        matcher = GetStartParenMatcher()
        matched_text = matcher.Match(None, remaining_string)
        remaining_string = remaining_string[len(matched_text):]
    return remaining_string





class SeparatedListFieldPlaceholder(ListFieldPlaceholder):

    def __init__(self, field_name, before_separator_placeholder=[], after__separator_placeholder=[]):
        super(SeparatedListFieldPlaceholder, self).__init__(
            field_name, before_placeholder=before_separator_placeholder,
            after_placeholder=after__separator_placeholder,
            exclude_first_before=True)


class DefaultSourceMatcher(SourceMatcher):
    """Class to generate the source for a node."""

    def __init__(self, node, expected_parts, starting_parens=None):
        super(DefaultSourceMatcher, self).__init__(node, starting_parens)
        previous_was_string = False
        # We validate that the expected parts does not contain two strings in
        # a row.
        for part in expected_parts:
            if not isinstance(part, fun_with_ast.placeholder_source_match.Placeholder):
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

    def __repr__(self):
        return ('DefaultSourceMatcher "{}" for node "{}" expecting to match "{}"'
                .format(super(DefaultSourceMatcher, self).__repr__(),
                        self.node,
                        pprint.pformat(self.expected_parts)))




# TODO: Add an indent placeholder that respects col_offset
def get_Add_expected_parts():
    return [TextPlaceholder(r'\+', '+')]


def get_alias_expected_parts():
    return [
        FieldPlaceholder('name'),
        FieldPlaceholder(
            'asname',
            before_placeholder=TextPlaceholder(r' *as *', ' as ')),
    ]


def get_And_expected_parts():
    return [TextPlaceholder(r'and')]


def get_arg_expected_parts():
    result = [FieldPlaceholder('arg')]
    return result


def get_arguments_expected_parts():
    return [
        ArgsDefaultsPlaceholder(
            TextPlaceholder(r'\s*,\s*', ', '),
            TextPlaceholder(r'\s*=\s*', '=')),
        FieldPlaceholder(
            'vararg',
            before_placeholder=TextPlaceholder(r'\s*,?\s*\*\s*', ', *')),
        FieldPlaceholder(
            'kwarg',
            before_placeholder=TextPlaceholder(r'\s*,?\s*\*\*\s*', ', **'))
    ]


def get_Assert_expected_parts():
    return [
        TextPlaceholder(r' *assert *', 'assert '),
        FieldPlaceholder('test'),
        FieldPlaceholder(
            'msg', before_placeholder=TextPlaceholder(r', *', ', ')),
        TextPlaceholder(r' *\n', '\n'),
    ]

def get_Assign_expected_parts():
    return [
        SeparatedListFieldPlaceholder('targets',   after__separator_placeholder=TextPlaceholder(r'\s*=\s*', '=')),
        FieldPlaceholder('value'),
        TextPlaceholder(r'[ \t]*\n?', ''),
    ]

# def get_Assign_expected_parts():
#     return [
#         TextPlaceholder(r'[ \t]*', ''),
#         SeparatedListFieldPlaceholder(
#             'targets', TextPlaceholder(r'\s*=\s*', ', ')),
#         TextPlaceholder(r'[ \t]*=[ \t]*', ' = '),
#         FieldPlaceholder('value'),
#         TextPlaceholder(r'\n', '\n')
#     ]


def get_Attribute_expected_parts():
    return [
        FieldPlaceholder('value'),
        TextPlaceholder(r'\s*\.\s*', '.'),
        FieldPlaceholder('attr')
    ]


def get_AugAssign_expected_parts():
    return [
        TextPlaceholder(r' *', ''),
        FieldPlaceholder('target'),
        TextPlaceholder(r' *', ' '),
        FieldPlaceholder('op'),
        TextPlaceholder(r'= *', '= '),
        FieldPlaceholder('value'),
        TextPlaceholder(r'\n', '\n')
    ]


# TODO: Handle parens better
def get_BinOp_expected_parts():
    return [
        FieldPlaceholder('left'),
        TextPlaceholder(r'\s*', ' '),
        FieldPlaceholder('op'),
        TextPlaceholder(r'\s*', ' '),
        FieldPlaceholder('right'),

    ]


def get_BitAnd_expected_parts():
    return [TextPlaceholder(r'&', '&')]


def get_BitOr_expected_parts():
    return [
        TextPlaceholder(r'\|', '|'),
    ]


def get_BitXor_expected_parts():
    return [
        TextPlaceholder(r'\^', '^'),
    ]


# TODO: Handle parens better
class BoolOpSourceMatcher(SourceMatcher):
    """Class to generate the source for an _ast.BoolOp node."""

    def __init__(self, node, starting_parens=None):
        super(BoolOpSourceMatcher, self).__init__(node, starting_parens)
        self.separator_placeholder = TextPlaceholder(r'\s*', ' ')
        self.matched_placeholders = []

    def GetSeparatorCopy(self):
        copy = self.separator_placeholder.Copy()
        self.matched_placeholders.append(copy)
        return copy

    def Match(self, string):
        remaining_string = self.MatchStartParens(string)

        elements = [self.node.values[0]]
        for value in self.node.values[1:]:
            elements.append(self.GetSeparatorCopy())
            elements.append(self.node.op)
            elements.append(self.GetSeparatorCopy())
            elements.append(value)

        parser = StringParser(remaining_string, elements, self.start_paren_matchers)
        matched_text = ''.join(parser.matched_substrings)
        remaining_string = parser.remaining_string

        self.MatchEndParen(remaining_string)

        return self.GetStartParenText() + matched_text + self.GetEndParenText()

    def GetSource(self):
        source_list = []
        if self.paren_wrapped:
            source_list.append(self.GetStartParenText())
        source_list.append(GetSource(self.node.values[0]))
        index = 0
        for value in self.node.values[1:]:
            source_list.append(_GetListDefault(
                self.matched_placeholders,
                index,
                self.separator_placeholder).GetSource(None))
            source_list.append(GetSource(self.node.op))
            index += 1
            source_list.append(_GetListDefault(
                self.matched_placeholders,
                index,
                self.separator_placeholder).GetSource(None))
            source_list.append(GetSource(value))
            index += 1
        if self.paren_wrapped:
            source_list.append(self.GetEndParenText())
        return ''.join(source_list)


def get_Break_expected_parts():
    return [TextPlaceholder(r' *break *\n', 'break\n')]


def get_Call_expected_parts():
    return [
        FieldPlaceholder('func'),
        TextPlaceholder(r'\(\s*', '('),
        ArgsKeywordsPlaceholder(
            TextPlaceholder(r'\s*,\s*', ', '),
            TextPlaceholder('')),
        FieldPlaceholder(
            'kwargs',
            before_placeholder=TextPlaceholder(r'\s*,?\s*\*\*', ', **')),
        TextPlaceholder(r'\s*,?\s*\)', ')'),
    ]


def get_ClassDef_expected_parts():
    return [
        ListFieldPlaceholder(
            'decorator_list',
            before_placeholder=TextPlaceholder('[ \t]*@', '@'),
            after_placeholder=TextPlaceholder(r'\n', '\n')),
        TextPlaceholder(r'[ \t]*class[ \t]*', 'class '),
        FieldPlaceholder('name'),
        TextPlaceholder(r'\(?\s*', '('),
        SeparatedListFieldPlaceholder(
            'bases', TextPlaceholder(r'\s*,\s*', ', ')),
        TextPlaceholder(r'\s*,?\s*\)?:\n', '):\n'),
        BodyPlaceholder('body')
    ]


def get_Compare_expected_parts():
    return [
        FieldPlaceholder('left'),
        TextPlaceholder(r'\s*', ' '),
        OpsComparatorsPlaceholder(
            TextPlaceholder(r'\s*', ' '),
            TextPlaceholder(r'\s*', ' '))
    ]


def get_comprehension_expected_parts():
    return [
        TextPlaceholder(r'\s*for\s*', 'for '),
        FieldPlaceholder('target'),
        TextPlaceholder(r'\s*in\s*', ' in '),
        FieldPlaceholder('iter'),
        ListFieldPlaceholder(
            'ifs',
            before_placeholder=TextPlaceholder(r'\s*if\s*', ' if '))
    ]


def get_Continue_expected_parts():
    return [TextPlaceholder(r' *continue\n')]


def get_Delete_expected_parts():
    return [
        TextPlaceholder(r' *del *'),
        ListFieldPlaceholder('targets'),
        TextPlaceholder(r'\n', '\n'),
    ]


def get_Dict_expected_parts():
    return [
        TextPlaceholder(r'\s*{\s*', '{'),
        KeysValuesPlaceholder(
            TextPlaceholder(r'\s*,\s*', ', '),
            TextPlaceholder(r'\s*:\s*', ': ')),
        TextPlaceholder(r'\s*,?\s*}', '}')
    ]


def get_Div_expected_parts():
    return [
        TextPlaceholder(r'/', '/'),
    ]


# TODO: Handle both types of k/v syntax
def get_DictComp_expected_parts():
    return [
        TextPlaceholder(r'\{\s*', '{'),
        FieldPlaceholder('key'),
        TextPlaceholder(r'\s*:\s*', ': '),
        FieldPlaceholder('value'),
        TextPlaceholder(r' *', ' '),
        ListFieldPlaceholder('generators'),
        TextPlaceholder(r'\s*\}', '}'),
    ]


def get_Eq_expected_parts():
    return [TextPlaceholder(r'==', '==')]


def get_ExceptHandler_expected_parts():
    return [
        TextPlaceholder(r'[ \t]*except:?[ \t]*', 'except '),
        FieldPlaceholder('type'),
        FieldPlaceholder(
            'name',
            before_placeholder=TextPlaceholder(r' *as *| *, *', ' as ')),
        TextPlaceholder(r'[ \t]*:?[ \t]*\n', ':\n'),
        BodyPlaceholder('body')
    ]


def get_Expr_expected_parts():
    return [
        TextPlaceholder(r' *', ''),
        FieldPlaceholder('value'),
        TextPlaceholder(r' *\n', '\n')
    ]


def get_FloorDiv_expected_parts():
    return [
        TextPlaceholder(r'//', '//'),
    ]


def get_For_expected_parts():
    return [
        TextPlaceholder(r'[ \t]*for[ \t]*', 'for '),
        FieldPlaceholder('target'),
        TextPlaceholder(r'[ \t]*in[ \t]*', ' in '),
        FieldPlaceholder('iter'),
        TextPlaceholder(r':\n', ':\n'),
        BodyPlaceholder('body'),
        BodyPlaceholder(
            'orelse',
            prefix_placeholder=TextPlaceholder(r' *else:\n', 'else:\n')),
    ]


def get_FunctionDef_expected_parts():
    return [
        BodyPlaceholder(
            'decorator_list',
            before_placeholder=TextPlaceholder('[ \t]*@', '@'),
            after_placeholder=TextPlaceholder(r'\n', '\n')),
        TextPlaceholder(r'[ \t]*def ', 'def '),
        FieldPlaceholder('name'),
        TextPlaceholder(r'\(\s*', '('),
        FieldPlaceholder('args'),
        TextPlaceholder(r'\s*,?\s*\):\n?', '):\n'),
        BodyPlaceholder('body')
    ]


def get_GeneratorExp_expected_parts():
    return [
        FieldPlaceholder('elt'),
        TextPlaceholder(r'\s*', ' '),
        ListFieldPlaceholder('generators'),
    ]


def get_Global_expected_parts():
    return [
        TextPlaceholder(r' *global *', 'global '),
        SeparatedListFieldPlaceholder(
            r'names',
            TextPlaceholder(r'\s*,\s*', ', ')),
        TextPlaceholder(r' *\n', '\n')
    ]


def get_Gt_expected_parts():
    return [TextPlaceholder(r'>', '>')]


def get_GtE_expected_parts():
    return [TextPlaceholder(r'>=', '>=')]


class IfSourceMatcher(SourceMatcher):
    """Class to generate the source for an _ast.If node."""

    def __init__(self, node, starting_parens=None):
        super(IfSourceMatcher, self).__init__(node, starting_parens)
        self.if_placeholder = TextPlaceholder(r' *if\s*', 'if ')
        self.test_placeholder = FieldPlaceholder('test')
        self.if_colon_placeholder = TextPlaceholder(r'[ \t]*:[ \t]*\n', ':\n')
        self.body_placeholder = BodyPlaceholder('body')
        self.else_placeholder = TextPlaceholder(r' *else:\s*', 'else:\n')
        self.orelse_placeholder = BodyPlaceholder('orelse')
        self.is_elif = False
        self.if_indent = 0

    def Match(self, string):
        self.if_indent = len(string) - len(string.lstrip())
        placeholder_list = [self.if_placeholder,
                            self.test_placeholder,
                            self.if_colon_placeholder,
                            self.body_placeholder]
        remaining_string = MatchPlaceholderList(
            string, self.node, placeholder_list)
        if not self.node.orelse:
            return string[:len(remaining_string)]
        else:
            # Handles the case of a blank line before an elif/else statement
            # Can't pass the "match_after" kwarg to self.body_placeholder,
            # because we don't want to match after if we don't have an else.
            while SyntaxFreeLine.MatchesStart(remaining_string):
                remaining_string, syntax_free_node = (
                    self.body_placeholder.MatchSyntaxFreeLine(remaining_string))
                self.node.body.append(syntax_free_node)
            if remaining_string.lstrip().startswith('elif'):
                self.is_elif = True
                indent = len(remaining_string) - len(remaining_string.lstrip())
                remaining_string = (remaining_string[:indent] +
                                    remaining_string[indent + 2:])
                # This is a hack to handle the fact that elif is a special case
                # BodyPlaceholder uses the indent of the other child statements
                # to match SyntaxFreeLines, which breaks in this case, because the
                # child isn't indented
                self.orelse_placeholder = ListFieldPlaceholder('orelse')
            else:
                remaining_string = MatchPlaceholder(
                    remaining_string, self.node, self.else_placeholder)
        remaining_string = self.orelse_placeholder.Match(
            self.node, remaining_string)
        if not remaining_string:
            return string
        return string[:len(remaining_string)]

    def GetSource(self):
        placeholder_list = [self.if_placeholder,
                            self.test_placeholder,
                            self.if_colon_placeholder,
                            self.body_placeholder]
        source_list = [p.GetSource(self.node) for p in placeholder_list]
        if not self.node.orelse:
            return ''.join(source_list)
        if (len(self.node.orelse) == 1 and
                isinstance(self.node.orelse[0], _ast.If) and
                self.is_elif):
            elif_source = GetSource(self.node.orelse[0])
            indent = len(elif_source) - len(elif_source.lstrip())
            source_list.append(elif_source[:indent] + 'el' + elif_source[indent:])
        else:
            if self.else_placeholder:
                source_list.append(self.else_placeholder.GetSource(self.node))
            else:
                source_list.append(' ' * self.if_indent)
                source_list.append('else:\n')
            source_list.append(self.orelse_placeholder.GetSource(self.node))
        return ''.join(source_list)


def get_IfExp_expected_parts():
    return [
        FieldPlaceholder('body'),
        TextPlaceholder(r'\s*if\s*', ' if '),
        FieldPlaceholder('test'),
        TextPlaceholder(r'\s*else\s*', ' else '),
        FieldPlaceholder('orelse'),
    ]


def get_Import_expected_parts():
    return [
        TextPlaceholder(r' *import ', 'import '),
        SeparatedListFieldPlaceholder(
            'names', TextPlaceholder('[ \t]*,[ \t]', ', ')),
        TextPlaceholder(r'\n', '\n')
    ]


def get_ImportFrom_expected_parts():
    return [
        TextPlaceholder(r'[ \t]*from ', 'from '),
        FieldPlaceholder('module'),
        TextPlaceholder(r' import '),
        SeparatedListFieldPlaceholder(
            'names', TextPlaceholder('[ \t]*,[ \t]', ', ')),
        TextPlaceholder(r'\n', '\n')
    ]


def get_In_expected_parts():
    return [TextPlaceholder(r'in', 'in')]


def get_Index_expected_parts():
    return [FieldPlaceholder(r'value')]


def get_Invert_expected_parts():
    return [TextPlaceholder(r'[ \t]*~', '~')]


def get_Is_expected_parts():
    return [TextPlaceholder(r'is', 'is')]


def get_IsNot_expected_parts():
    return [TextPlaceholder(r'is *not', 'is not')]


def get_keyword_expected_parts():
    return [
        FieldPlaceholder('arg'),
        TextPlaceholder(r'\s*=\s*', '='),
        FieldPlaceholder('value'),
    ]


def get_Lambda_expected_parts():
    return [
        TextPlaceholder(r'lambda\s*', 'lambda '),
        FieldPlaceholder('args'),
        TextPlaceholder(r'\s*:\s*', ': '),
        FieldPlaceholder('body'),
    ]


def get_List_expected_parts():
    return [
        TextPlaceholder(r'\[\s*', '['),
        SeparatedListFieldPlaceholder(
            'elts', TextPlaceholder(r'\s*,\s*', ', ')),
        TextPlaceholder(r'\s*,?\s*\]', ']')]


def get_ListComp_expected_parts():
    return [
        TextPlaceholder(r'\[\s*', '['),
        FieldPlaceholder('elt'),
        TextPlaceholder(r' *', ' '),
        ListFieldPlaceholder('generators'),
        TextPlaceholder(r'\s*\]', ']'),
    ]


def get_LShift_expected_parts():
    return [
        TextPlaceholder(r'<<', '<<'),
    ]


def get_Lt_expected_parts():
    return [TextPlaceholder(r'<', '<')]


def get_LtE_expected_parts():
    return [TextPlaceholder(r'<=', '<=')]


def get_Mod_expected_parts():
    return [TextPlaceholder(r'%')]


def get_Module_expected_parts():
    return [BodyPlaceholder('body')]


def get_Mult_expected_parts():
    return [TextPlaceholder(r'\*', '*')]


def get_Name_expected_parts():
    return [TextPlaceholder(r'[ \t]*', ''),
            FieldPlaceholder('id'),
            TextPlaceholder(r'([ \t]*)', '')]
#            TextPlaceholder(r'[ \t]+|[ \t]*#.*', '')]
#    return [FieldPlaceholder('id')]


def get_NotEq_expected_parts():
    return [TextPlaceholder(r'!=')]


def get_Not_expected_parts():
    return [TextPlaceholder(r'not', 'not')]


def get_NotIn_expected_parts():
    return [TextPlaceholder(r'not *in', 'not in')]



def get_Or_expected_parts():
    return [TextPlaceholder(r'or')]


def get_Pass_expected_parts():
    return [TextPlaceholder(r'[ \t]*pass[ \t]*#*.*\n*', 'pass')]


def get_Pow_expected_parts():
    return [
        TextPlaceholder(r'\*\*', '**'),
    ]


# TODO: Support non-nl syntax
def get_Print_expected_parts():
    return [
        TextPlaceholder(r' *print *', 'print '),
        FieldPlaceholder(
            'dest',
            before_placeholder=TextPlaceholder(r'>>', '>>')),
        ListFieldPlaceholder(
            r'values',
            TextPlaceholder(r'\s*,?\s*', ', ')),
        TextPlaceholder(r' *,? *\n', '\n')
    ]

def get_JoinedStr_expected_parts():
    return [
        TextPlaceholder(r'f\'', 'f\''),
        ListFieldPlaceholder(
            r'values',
            after_placeholder=TextPlaceholder(r'\s*,?\s*', ', ')),
        TextPlaceholder(r'\'', '\'')
    ]


def get_Raise_expected_parts():
    return [
        TextPlaceholder(r'[ \t]*raise[ \t]*', 'raise '),
        FieldPlaceholder('type'),
        TextPlaceholder(r'\n', '\n'),
    ]


def get_Return_expected_parts():
    return [
        TextPlaceholder(r'[ \t]*return[ \t]*', 'return '),
        FieldPlaceholder('value'),
#        TextPlaceholder(r'\n', '\n'),
    ]


def get_RShift_expected_parts():
    return [
        TextPlaceholder(r'>>', '>>'),
    ]


def get_Set_expected_parts():
    return [
        TextPlaceholder(r'\{\s*', '{'),
        SeparatedListFieldPlaceholder(
            'elts', TextPlaceholder(r'\s*,\s*', ', ')),
        TextPlaceholder(r'\s*\}', '}'),
    ]


def get_SetComp_expected_parts():
    return [
        TextPlaceholder(r'\{\s*', '{'),
        FieldPlaceholder('elt'),
        TextPlaceholder(r' *', ' '),
        ListFieldPlaceholder('generators'),
        TextPlaceholder(r'\s*\}', '}'),
    ]


def get_Slice_expected_parts():
    return [
        FieldPlaceholder('lower'),
        TextPlaceholder(r'\s*:?\s*', ':'),
        FieldPlaceholder('upper'),
        TextPlaceholder(r'\s*:?\s*', ':'),
        FieldPlaceholder('step'),
    ]






def get_Sub_expected_parts():
    return [
        TextPlaceholder(r'\-', '-'),
    ]


def get_Subscript_expected_parts():
    return [
        FieldPlaceholder('value'),
        TextPlaceholder(r'\s*\[\s*', '['),
        FieldPlaceholder('slice'),
        TextPlaceholder(r'\s*\]', ']'),
    ]


def get_SyntaxFreeLine_expected_parts():
    return [FieldPlaceholder('full_line'),
            TextPlaceholder(r'\n', '\n')]

def get_Comment_expected_parts():
    return [TextPlaceholder(r'#.*', '#')]


class TupleSourceMatcher(DefaultSourceMatcher):
    """Source matcher for _ast.Tuple nodes."""

    def __init__(self, node, starting_parens=None):
        expected_parts = [
            TextPlaceholder(r'\s*\(', ''),
            SeparatedListFieldPlaceholder(
                'elts', before_separator_placeholder=TextPlaceholder(r'[ \t]*,[ \t]*', ',')),
            TextPlaceholder(r'\s*,?\s*\)[ \t]*(#\S*)*', ')')
        ]
        super(TupleSourceMatcher, self).__init__(
            node, expected_parts, starting_parens)

    def Match(self, string):
        matched_text = super(TupleSourceMatcher, self).Match(string)
        return matched_text
#        if not self.paren_wrapped:
#            matched_text = matched_text.rstrip()
#            return super(TupleSourceMatcher, self).Match(matched_text)

    def MatchStartParens(self, remaining_string):
        return remaining_string
        # if remaining_string.startswith('(('):
        #    raise NotImplementedError('Currently not supported')
        # if remaining_string.startswith('('):
        #    return remaining_string
        # raise ValueError('Tuple does not start with (')

def get_TryExcept_expected_parts():
    return [
        TextPlaceholder(r'[ \t]*try:[ \t]*\n', 'try:\n'),
        BodyPlaceholder('body', match_after=True),
        ListFieldPlaceholder('handlers'),
        BodyPlaceholder(
            'orelse',
            prefix_placeholder=TextPlaceholder(r'[ \t]*else:\n', 'else:\n')),
        BodyPlaceholder(
            'finalbody',
            prefix_placeholder= TextPlaceholder(r'[ \t]*finally:[ \t]*\n', 'finally:\n'))
    ]


# python 3 matching for ast.Constant


class TryFinallySourceMatcher(DefaultSourceMatcher):

    def __init__(self, node, starting_parens=None):
        expected_parts = [
            BodyPlaceholder('body', match_after=True),
            TextPlaceholder(r'[ \t]*finally:[ \t]*\n', 'finally:\n'),
            BodyPlaceholder('finalbody'),
        ]
        super(TryFinallySourceMatcher, self).__init__(
            node, expected_parts, starting_parens)
        self.optional_try = TextPlaceholder(r'[ \t]*try:[ \t]*\n', 'try:\n')

    def Match(self, string):
        remaining_string = string
        if not isinstance(self.node.body[0], _ast.Try):
            remaining_string = MatchPlaceholder(remaining_string, None, self.optional_try)
        return super(TryFinallySourceMatcher, self).Match(remaining_string)

    def GetSource(self):
        source_start = ''
        if not isinstance(self.node.body[0], _ast.TryExcept):
            source_start = self.optional_try.GetSource(None)
        return source_start + super(TryFinallySourceMatcher, self).GetSource()


def get_UAdd_expected_parts():
    return [
        TextPlaceholder(r'\+', '+'),
    ]


def get_UnaryOp_expected_parts():
    return [
        FieldPlaceholder('op'),
        TextPlaceholder(r' *', ' '),
        FieldPlaceholder('operand'),
    ]


def get_USub_expected_parts():
    return [
        TextPlaceholder(r'-', '-'),
    ]


def get_While_expected_parts():
    return [
        TextPlaceholder(r'[ \t]*while[ \t]*', 'while '),
        FieldPlaceholder('test'),
        TextPlaceholder(r'[ \t]*:[ \t]*\n', ':\n'),
        BodyPlaceholder('body'),
    ]


class WithItemSourceMatcher(SourceMatcher):
    def __init__(self, node, starting_parens=None):
        super(WithItemSourceMatcher, self).__init__(node, starting_parens)
        self.context_expr = FieldPlaceholder('context_expr')
        self.optional_vars = FieldPlaceholder(
            'optional_vars',
            before_placeholder=TextPlaceholder(r' *as *', ' as '))

        #self.compound_separator = TextPlaceholder(r'\s*,\s*', ', ')

    def Match(self, string):
        #    if 'as' not in string:
        #      return MatchPlaceholder(string, self.node, self.context_expr)
        placeholder_list = [self.context_expr,
                            self.optional_vars]
        remaining_string = MatchPlaceholderList(
            string, self.node, placeholder_list)

        if not remaining_string:
            return string
        return string[:len(remaining_string)]

    def GetSource(self):
        source_list = []
        placeholder_list = [self.context_expr,
                            self.optional_vars]
        source_list = [p.GetSource(self.node) for p in placeholder_list]
        return ''.join(source_list)



# def get_withitem_expected_parts():
#  return [
#    FieldPlaceholder('context_expr'),
#    TextPlaceholder(r' *as *', ' as '),
#    FieldPlaceholder('optional_vars'),
#  ]


class WithSourceMatcher(SourceMatcher):
    """Class to generate the source for an _ast.With node."""

    def __init__(self, node, starting_parens=None):
        super(WithSourceMatcher, self).__init__(node, starting_parens)
        self.with_placeholder = TextPlaceholder(r' *(with)? *', 'with ')
        self.withitems_placeholder = SeparatedListFieldPlaceholder('items', before_separator_placeholder=TextPlaceholder(r', *', ', '))
        #    self.context_expr = FieldPlaceholder('context_expr')
        #    self.optional_vars = FieldPlaceholder(
        #        'optional_vars',
        #        before_placeholder=TextPlaceholder(r' *as *', ' as '))
#        self.compound_separator = TextPlaceholder(r'\s*,\s*', ', ')
        self.colon_placeholder = TextPlaceholder(r':\n?', ':\n')
        self.body_placeholder = BodyPlaceholder('body')
        self.is_compound_with = False
        self.starting_with = True

    def Match(self, string):
        if string.lstrip().startswith('with'):
            self.starting_with = True
        placeholder_list = [self.with_placeholder,
                            self.withitems_placeholder]
        remaining_string = MatchPlaceholderList(
            string, self.node, placeholder_list)
        if remaining_string.lstrip().startswith(','):
            self.is_compound_with = True
            placeholder_list = [self.compound_separator,
                                self.body_placeholder]
            remaining_string = MatchPlaceholderList(
                remaining_string, self.node, placeholder_list)
        else:
            placeholder_list = [self.colon_placeholder,
                                self.body_placeholder]
            remaining_string = MatchPlaceholderList(
                remaining_string, self.node, placeholder_list)

        if not remaining_string:
            return string
        return string[:len(remaining_string)]

    def GetSource(self):
        placeholder_list = []
        if self.starting_with:
            placeholder_list.append(self.with_placeholder)
        placeholder_list.append(self.withitems_placeholder)
#        placeholder_list.append(self.optional_vars)
        if (self.is_compound_with and
                isinstance(self.node.body[0], _ast.With)):
            if not hasattr(self.node.body[0], 'matcher'):
                # Triggers attaching a matcher. We don't act like an stmt,
                # so we can assume no indent.
                GetSource(self.node.body[0], assume_no_indent=True)
            # If we're part of a compound with, we want to make
            # sure the initial "with" of the body isn't included
            self.node.body[0].matcher.starting_with = False
            placeholder_list.append(self.compound_separator)
        else:
            # If we're not a compound with, we expect the colon
            placeholder_list.append(self.colon_placeholder)
        placeholder_list.append(self.body_placeholder)

        source_list = [p.GetSource(self.node) for p in placeholder_list]
        return ''.join(source_list)


def get_Yield_expected_parts():
    return [
        TextPlaceholder(r'[ \t]*yield[ \t]*', 'yield '),
        FieldPlaceholder('value'),
    ]


# A mapping of node_type: expected_parts
# _matchers = LazyDict({
#     _ast.Add: (get_Add_expected_parts),
#     _ast.alias: (get_alias_expected_parts),
#     _ast.And: (get_And_expected_parts),
#     _ast.Assert: (get_Assert_expected_parts),
#     _ast.Assign: (get_Assign_expected_parts),
#     _ast.Attribute: (get_Attribute_expected_parts),
#     _ast.AugAssign: (get_AugAssign_expected_parts),
#     _ast.arguments: (get_arguments_expected_parts),
#     _ast.arg: (get_arg_expected_parts),
#     _ast.BinOp: (get_BinOp_expected_parts),
#     _ast.BitAnd: (get_BitAnd_expected_parts),
#     _ast.BitOr: (get_BitOr_expected_parts),
#     _ast.BitXor: (get_BitXor_expected_parts),
#     _ast.BoolOp: (BoolOpSourceMatcher),
#     _ast.Break: (get_Break_expected_parts),
#     _ast.Call: (get_Call_expected_parts),
#     _ast.ClassDef: (get_ClassDef_expected_parts),
#     _ast.Compare: (get_Compare_expected_parts),
#     _ast.comprehension: (get_comprehension_expected_parts),
#     _ast.Continue: (get_Continue_expected_parts),
#     _ast.Delete: (get_Delete_expected_parts),
#     _ast.Dict: (get_Dict_expected_parts),
#     _ast.DictComp: (get_DictComp_expected_parts),
#     _ast.Div: (get_Div_expected_parts),
#     _ast.Eq: (get_Eq_expected_parts),
#     _ast.Expr: (get_Expr_expected_parts),
#     _ast.ExceptHandler: (get_ExceptHandler_expected_parts),
#     _ast.FloorDiv: (get_FloorDiv_expected_parts),
#     _ast.For: (get_For_expected_parts),
#     _ast.FunctionDef: (get_FunctionDef_expected_parts),
#     _ast.GeneratorExp: (get_GeneratorExp_expected_parts),
#     _ast.Global: (get_Global_expected_parts),
#     _ast.Gt: (get_Gt_expected_parts),
#     _ast.GtE: (get_GtE_expected_parts),
#     _ast.If: (IfSourceMatcher),
#     _ast.IfExp: (get_IfExp_expected_parts),
#     _ast.Import: (get_Import_expected_parts),
#     _ast.ImportFrom: (get_ImportFrom_expected_parts),
#     _ast.In: (get_In_expected_parts),
#     #    _ast.Index: get_Index_expected_parts),
#     _ast.Invert: (get_Invert_expected_parts),
#     _ast.Is: (get_Is_expected_parts),
#     _ast.IsNot: (get_IsNot_expected_parts),
#     _ast.keyword: (get_keyword_expected_parts),
#     _ast.Lambda: (get_Lambda_expected_parts),
#     _ast.List: (get_List_expected_parts),
#     _ast.ListComp: (get_ListComp_expected_parts),
#     _ast.LShift: (get_LShift_expected_parts),
#     _ast.Lt: (get_Lt_expected_parts),
#     _ast.LtE: (get_LtE_expected_parts),
#     _ast.Mod: (get_Mod_expected_parts),
#     _ast.Module: (get_Module_expected_parts),
#     _ast.Mult: (get_Mult_expected_parts),
#     _ast.Name: (get_Name_expected_parts),
#     _ast.Not: (get_Not_expected_parts),
#     _ast.NotIn: (get_NotIn_expected_parts),
#     _ast.NotEq: (get_NotEq_expected_parts),
#     #    _ast.Num: NumSourceMatcher),
#     _ast.Or: (get_Or_expected_parts),
#     _ast.Pass: (get_Pass_expected_parts),
#     _ast.Pow: (get_Pow_expected_parts),
#     #    _ast.Print: get_Print_expected_parts),
#     _ast.Raise: (get_Raise_expected_parts),
#     _ast.Return: (get_Return_expected_parts),
#     _ast.RShift: (get_RShift_expected_parts),
#     _ast.Slice: (get_Slice_expected_parts),
#     _ast.Sub: (get_Sub_expected_parts),
#     _ast.Set: (get_Set_expected_parts),
#     _ast.SetComp: (get_SetComp_expected_parts),
#     _ast.Subscript: (get_Subscript_expected_parts),
#     #    _ast.Str: (StrSourceMatcher),
#     _ast.Constant: (ConstantSourceMatcher),
#     SyntaxFreeLine: (get_SyntaxFreeLine_expected_parts),
#     Comment: (get_Comment_expected_parts),
#     _ast.Tuple: (TupleSourceMatcher),
#     #    _ast.TryExcept: get_TryExcept_expected_parts),
#     #    _ast.Try: TryFinallySourceMatcher),
#     _ast.JoinedStr: (get_JoinedStr_expected_parts),
#     _ast.Try: (get_TryExcept_expected_parts),
#     _ast.UAdd: (get_UAdd_expected_parts),
#     _ast.UnaryOp: (get_UnaryOp_expected_parts),
#     _ast.USub: (get_USub_expected_parts),
#     _ast.While: (get_While_expected_parts),
#     _ast.With: (WithSourceMatcher),
#     _ast.withitem: (WithItemSourceMatcher),
#     _ast.Yield: (get_Yield_expected_parts)
# })
