import fun_with_ast.dynamic_matcher
import fun_with_ast.node_tree_util



def FixSourceIndentation(module_node, node_to_fix, starting_parens=None):
    if starting_parens is None:
        starting_parens = []
    default_source = node_to_fix.matcher.GetSource()
    node_to_fix.matcher = dynamic_matcher.GetDynamicMatcher(node_to_fix, starting_parens)
    starting_indent = '  ' * node_tree_util.GetIndentLevel(
        module_node, node_to_fix)
    node_to_fix.matcher.Match(starting_indent + default_source)

def GetDefaultQuoteType():
    return '"'

def _GetListDefault(l, index, default):
    if index < len(l):
        return l[index]
    else:
        return default.Copy()

def _IsBackslashEscapedQuote(string, quote_index):
    """Checks if the quote at the given index is backslash escaped."""
    num_preceding_backslashes = 0
    for char in reversed(string[:quote_index]):
        if char == '\\':
            num_preceding_backslashes += 1
        else:
            break
    return num_preceding_backslashes % 2 == 1

def _FindQuoteEnd(string, quote_type):
    """Recursively finds the ending index of a quote.

    Args:
      string: The string to search inside of.
      quote_type: The quote type we're looking for.

    Returns:
      The index of the end of the first quote.

    The method works by attempting to find the first instance of the end of
    the quote, then recursing if it isn't valid. If -1 is returned at any time,
    we can't find the end, and we return -1.
    """
    trial_index = string.find(quote_type)
    if trial_index == -1:
        return -1
    elif not _IsBackslashEscapedQuote(string, trial_index):
        return trial_index
    else:
        new_start = trial_index + 1
        rest_index = _FindQuoteEnd(string[new_start:], quote_type)
        if rest_index == -1:
            return -1
        else:  # Return the recursive sum
            return new_start + rest_index
