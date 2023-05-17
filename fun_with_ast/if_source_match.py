import _ast

from fun_with_ast.source_matchers.body import BodyPlaceholder
from placeholders.composite_placeholder_source_match import FieldPlaceholder
from placeholders.list_placeholder_source_match import ListFieldPlaceholder
from fun_with_ast.manipulate_node.create_node import SyntaxFreeLine
from fun_with_ast.get_source import GetSource
from fun_with_ast.source_matchers.base_matcher import SourceMatcher, MatchPlaceholderList, MatchPlaceholder
from fun_with_ast.placeholders.text_placeholder import TextPlaceholder


class IfSourceMatcher(SourceMatcher):
    """Class to generate the source for an _ast.If node."""

    def __init__(self, node, starting_parens=None, parent=None):
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
