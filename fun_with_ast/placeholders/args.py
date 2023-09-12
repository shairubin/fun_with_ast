import re

from fun_with_ast.placeholders.list_placeholder import SeparatedListFieldPlaceholder
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher

from fun_with_ast.manipulate_node.call_args_node import CallArgs
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
        self.args_matcher = None
        self.use_default_matcher = True # TODO: remove this feature flag
    def _match(self, node, string):

        if self.use_default_matcher == True:
            default_matcher_result = self._use_default_matcher(node, string)
            return default_matcher_result
        else:
            raise ValueError('old implementation not supported anymore')
    def GetElements(self, node):
        if self.use_default_matcher == True and self.args_matcher:
            elements = []
            elements.extend(self.args_matcher.start_paren_matchers)
            elements.extend(self.args_matcher.expected_parts)
            elements.extend(self.args_matcher.end_paren_matchers)
            elements.extend(self.args_matcher.EOL_comment_matcher)
            return elements
        elif self.use_default_matcher == True and not self.args_matcher:
            parts = self._get_parts_for_default_matcher(0, node, '')
            start_paren = TextPlaceholder(r'\(\s*', '(')
            end_paren = TextPlaceholder(r'\s*,?\s*\)', ')')
            parts.insert(0, start_paren)
            parts.append(end_paren)
            return parts
        else:
            raise ValueError('old implementation not supported anymore')

    def _use_default_matcher(self, node, string):
        arg_index = len(node.args)
        args_node = CallArgs(node.args, node.keywords, node)
        node.keywords = args_node.keywords # not nice
        parts = self._get_parts_for_default_matcher(arg_index, node, string)
        self.args_matcher = GetDynamicMatcher(args_node, parts_in=parts)
        matched_string = self.args_matcher._match(string)
        return matched_string

    def _get_parts_for_default_matcher(self, arg_index, node, string):
        parts = []
        args_seperator_placeholder = TextPlaceholder(r'(\s*,\s*)?([ \t]*#.*\n*[ \t]*)*', default='', no_transform=True )
        exclude_last_after = self._should_exclude_last_after(string)
        parts.append(SeparatedListFieldPlaceholder(r'args',
                                                    after__separator_placeholder=args_seperator_placeholder,
                                                    exclude_last_after=exclude_last_after))
        if node.keywords:
            if node.args:
                parts.append(args_seperator_placeholder)
            parts.append(SeparatedListFieldPlaceholder(r'keywords', after__separator_placeholder=args_seperator_placeholder,
                                                        exclude_last_after=exclude_last_after))
        if getattr(node, 'starargs', False):
            ValueError('This should not happen in python 3.10; starred args are part of args')
            parts.append(self.stararg_separator)
            parts.append(NodePlaceholder(node.starargs))
            if node.keywords:
                arg_seperator = self._GetArgSeparator(arg_index)
                parts.append(arg_seperator)
                arg_index += 1
            for index, arg in enumerate(node.keywords):
                parts.append(NodePlaceholder(arg))
                if index != len(node.keywords) - 1:
                    parts.append(self._GetArgSeparator(arg_index))
                    arg_index += 1
        return parts

    def _should_exclude_last_after(self, string):
        parens_pairs = self._find_parens(string)
        if not parens_pairs:
            return True
        first_pair = parens_pairs[0]
        current_string = string[:first_pair[1]+1]
        if re.search(r'[ \t]*,[ \t\n]*\)$', current_string):
            return False
        if re.search(r'(\s*,\s*)?([ \t]*#.*\n*[ \t]*)', current_string):
            return False
        return  True

    def _find_parens(self, s): # from stack overflow
        toret = {}
        pstack = []

        for i, c in enumerate(s):
            if c == '(':
                pstack.append(i)
            elif c == ')':
                if len(pstack) == 0:
                    break
                toret[pstack.pop()] = i

        if len(pstack) > 0:
            raise IndexError("No matching opening parens at: " + str(pstack.pop()))
        result = [(x,y) for x,y in toret.items()]
        result.sort()
        return result

class OpsComparatorsPlaceholder(ArgsDefaultsPlaceholder):

    def _GetArgsKwargs(self, node):
        return [], list(zip(node.ops, node.comparators))

