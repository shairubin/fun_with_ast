import _ast

from fun_with_ast.dynamic_matcher import GetDynamicMatcher


def GetSource(field, text=None, starting_parens=None, assume_no_indent=False):
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
    if isinstance(field, int):
        return str(field)
    if hasattr(field, 'matcher') and field.matcher:
        return field.matcher.GetSource()
    else:
        field.matcher = GetDynamicMatcher(field, starting_parens)
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

        source_code = field.matcher.GetSource()
        return source_code
