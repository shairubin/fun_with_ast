import dynamic_matcher
import node_tree_util


def FixSourceIndentation(
        module_node, node_to_fix, starting_parens=None):
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
