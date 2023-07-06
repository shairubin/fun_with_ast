from fun_with_ast.source_matchers.base_matcher import SourceMatcher

from fun_with_ast.placeholders.composite import CompositePlaceholder
from fun_with_ast.placeholders.node import NodePlaceholder
from fun_with_ast.placeholders.text import TextPlaceholder


class ArgsDefaultsPlaceholder(CompositePlaceholder):
    """Placeholder to handle args and defaults for _ast.argument.

    These fields behave differently than most other fields and therefore
    don't fall into any of the other placeholders. Therefore, we have to define
    a custom placeholder.
    """

    def __init__(self, arg_separator_placeholder, kwarg_separator_placeholder):
        super(ArgsDefaultsPlaceholder, self).__init__()
        self.arg_separator_placeholder = arg_separator_placeholder
        self.kwarg_separator_placeholder = kwarg_separator_placeholder
        self.arg_separators = []
        self.kwarg_separators = []

    def _GetArgSeparator(self, index):
        if index < len(self.arg_separators):
            return self.arg_separators[index]
        new_placeholder = self.arg_separator_placeholder.Copy()
        self.arg_separators.append(new_placeholder)
        return new_placeholder

    def _GetKwargSeparator(self, index):
        if index < len(self.kwarg_separators):
            return self.kwarg_separators[index]
        new_placeholder = self.kwarg_separator_placeholder.Copy()
        self.kwarg_separators.append(new_placeholder)
        return new_placeholder

    def _GetArgsKwargs(self, node):
        kwargs = list(zip(node.args[len(node.args) - len(node.defaults):], node.defaults))
        args = node.args[:-len(kwargs)] if kwargs else node.args
        return args, kwargs

    def GetElements(self, node):
        """Gets the basic elements of this composite placeholder."""
        args, kwargs = self._GetArgsKwargs(node)
        elements = []
        arg_index = 0
        kwarg_index = 0
        for index, arg in enumerate(args):
            elements.append(NodePlaceholder(arg))
            if index is not len(args) - 1 or kwargs:
                elements.append(self._GetArgSeparator(arg_index))
                arg_index += 1
        for index, (key, val) in enumerate(kwargs):
            elements.append(NodePlaceholder(key))
            elements.append(self._GetKwargSeparator(kwarg_index))
            kwarg_index += 1
            elements.append(NodePlaceholder(val))
            if index is not len(kwargs) - 1:
                elements.append(self._GetArgSeparator(arg_index))
                arg_index += 1
        return elements

    def __repr__(self):
        return ('ArgsDefaultsPlaceholder separating args with "{}" '
                'and kwargs with "{}"'
                .format(self.arg_separator_placeholder,
                        self.kwarg_separator_placeholder))


class KeysValuesPlaceholder(ArgsDefaultsPlaceholder):

    def _GetArgsKwargs(self, node):
        return [], list(zip(node.keys, node.values))


class ArgsKeywordsPlaceholder(ArgsDefaultsPlaceholder):

    def __init__(self, arg_separator_placeholder, kwarg_separator_placeholder):
        super(ArgsKeywordsPlaceholder, self).__init__(
            arg_separator_placeholder, kwarg_separator_placeholder)
        self.stararg_separator = TextPlaceholder(r'\s*,?\s*\*', ', *')
        self.start_paren_matchers = []
    def _match(self, node, string):
        if not string.startswith('('):
            raise ValueError('string for arguments _match does not start with string')
        #remaing_string = SourceMatcher.parentheses_stack.MatchStartParens(self,string)
        remaing_string = super()._match(node, string)
        #remaing_string = SourceMatcher.parentheses_stack.MatchEndParens(self,remaing_string)
        return remaing_string
    def GetElements(self, node):
        """Gets the basic elements of this composite placeholder."""
        args = node.args or []
        keywords = node.keywords or []
        elements = []
        arg_index = 0
        for index, arg in enumerate(args):
            elements.append(NodePlaceholder(arg))
            if index != len(args) - 1 or keywords:
                elements.append(self._GetArgSeparator(arg_index))
                arg_index += 1
        if getattr(node, 'starargs', False):
            elements.append(self.stararg_separator)
            elements.append(NodePlaceholder(node.starargs))
            if keywords:
                elements.append(self._GetArgSeparator(arg_index))
                arg_index += 1
        for index, arg in enumerate(keywords):
            elements.append(NodePlaceholder(arg))
            if index != len(keywords) - 1:
                elements.append(self._GetArgSeparator(arg_index))
                arg_index += 1
        #if not elements:
        #     parens = TextPlaceholder(r'\(\s*\)', '()')
        #     elements.append(parens)
        # elif not node.args and  getattr(node, 'starargs', False):
        #     start_paren = TextPlaceholder(r'\(\s*', '(')
        #     end_paren = TextPlaceholder(r'\s*,?\s*\)', ')')
        #     elements.insert(0,start_paren)
        #     elements.append(end_paren)
        # else:
        #     ValueError('Missing rule for adding parentheses to arguments')
        start_paren = TextPlaceholder(r'\(\s*', '(')
        end_paren = TextPlaceholder(r'\s*,?\s*\)', ')')
        elements.insert(0,start_paren)
        elements.append(end_paren)

        return elements


class OpsComparatorsPlaceholder(ArgsDefaultsPlaceholder):

    def _GetArgsKwargs(self, node):
        return [], list(zip(node.ops, node.comparators))
