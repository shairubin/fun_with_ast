from fun_with_ast.get_source import GetSource


class BodyManipulator:
    def _add_newlines(self, body_block):
        for node in body_block:
            node_source = node.matcher.GetSource()
            if node_source.endswith("\n"):
                continue
            node.matcher.add_newline_to_source()

    def inject_to_body(self, body_block_to_manipulate, node_to_inject, index):
        ident = self._get_node_indentation(body_block_to_manipulate)
        source_to_inject = GetSource(node_to_inject, assume_no_indent=True)
        node_to_inject.matcher.FixIndentation(ident)
        body_block_to_manipulate.insert(index, node_to_inject)
        self._add_newlines(body_block_to_manipulate)

    def _get_node_indentation(self, node):
        ident = 0
        for stmt in node:
            stmt_ident = stmt.col_offset
            if stmt_ident > ident and ident == 0:
                ident = stmt_ident
            elif stmt_ident != ident and ident != 0:
                raise ValueError('illegal ident')
        return ident
