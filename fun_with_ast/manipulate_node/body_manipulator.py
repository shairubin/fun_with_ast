from fun_with_ast.common_utils.node_tree_util import IsEmptyModule
from fun_with_ast.get_source import GetSource


class BodyManipulator:

    def __init__(self, body_block):
        self.body_block = body_block
    def _add_newlines(self):
        for node in self.body_block:
            node_source = node.matcher.GetSource()
            if node_source.endswith("\n"):
                continue
            node.matcher.add_newline_to_source()

    def inject_node(self,node_to_inject, index):
        if IsEmptyModule(node_to_inject):
            return
        ident = self._get_indentation()
        source_to_inject = GetSource(node_to_inject, assume_no_indent=True)
        node_to_inject.matcher.FixIndentation(ident)
        self.body_block.insert(index, node_to_inject)
        self._add_newlines()

    def get_source(self):
        raise NotImplementedError('get_source not implemented yet')

    def _get_indentation(self):
        ident = 0
        for stmt in self.body_block:
            stmt_ident = stmt.col_offset
            if stmt_ident > ident and ident == 0:
                ident = stmt_ident
            elif stmt_ident != ident and ident != 0:
                raise ValueError('illegal ident')
        return ident
