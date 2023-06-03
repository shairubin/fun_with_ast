import _ast
import ast

from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.common_utils.utils_source_match import FixSourceIndentation



def GetSource(field, text=None, starting_parens=None, assume_no_indent=False,
              parent_node=None, assume_elif=False):
    """Gets the source corresponding with a given field.

    If the node is not a string or a node with a .matcher function,
    this will get the matcher for the node, attach the matcher, and
    match the text provided. If no text is provided, it will rely on defaults.

    Args:
      field: {str|_ast.AST} The field we want the source from.
      text: {str} The text to match if a matcher doesn't exist.
      starting_parens: {[TextPlaceholder]} The list of parens that the field
          starts with.
      assume_no_indent: {bool} True if we can assume the node isn't indented.
          Used for things like new nodes that aren't yet in a module.

    Returns:
      A string, representing the source code for the node.

    Raises:
      ValueError: When passing in a stmt node that has no string or module_node.
          This is an error because we have no idea how much to indent it.
    """
    if field is None:
        return ''
    if starting_parens is None:
        starting_parens = []
    if isinstance(field, str):
        return field
    if isinstance(field, bool):
        return str(field)
    if isinstance(field, int):
        return _str_from_int(field, parent_node, text)
    if hasattr(field, 'matcher') and field.matcher:
        return field.matcher.GetSource()
    else:
        field.matcher = GetDynamicMatcher(field, starting_parens, parent_node=parent_node)
        _match_text(assume_no_indent, field, text)
        _set_elif(assume_elif, field)
        source_code = field.matcher.GetSource()
        return source_code


def _set_elif(assume_elif, field):
    if isinstance(field, _ast.If):
        if (assume_elif or (hasattr(field, 'is_alif') and field.is_alif)):
            field.matcher.is_elif = assume_elif


def _match_text(assume_no_indent, field, text):
    if text:
        field.matcher.Match(text)
    # TODO: Fix this to work with lambdas
    elif isinstance(field, _ast.stmt) and not assume_no_indent:
        if not hasattr(field, 'module_node'):
            raise ValueError(
                'No text was provided, and we try to get source from node {} which'
                'is a statement, so it must have a .module_node field defined. '
                'To add this automatically, call ast_annotate.AddBasicAnnotations'
                .format(field))
        FixSourceIndentation(field.module_node, field)


def _guess_base_from_string(string, field):
    if string.startswith('0x'):
        return hex(field)
    if string.startswith('0b'):
        return bin(field)
    if string.startswith('0o'):
        return oct(field)
    return str(field)


def _str_from_int(field, parent_node, string):
    if parent_node is None:
        return str(field)
    if not isinstance(parent_node, (ast.Constant, ast.Assign)):
        raise ValueError('not a Constant node, not supported')
    if not hasattr(parent_node, 'base') or parent_node is None:
        if string is not None:
            return _guess_base_from_string(string, field)
        else:
            return str(field)
    if parent_node.base == 2:
        return bin(field)
    if parent_node.base == 8:
        return oct(field)
    if parent_node.base == 16:
        return hex(field)
    return str(field)

