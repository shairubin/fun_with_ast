from fun_with_ast.utils_source_match import FixSourceIndentation
from fun_with_ast.get_source import GetSource



class ManipulateIfNode():
    def __init__(self, node):
        self.node = node

    def add_nodes_to_body(self, nodes: list, location: int):
        if location > len(self.node.body):
            raise ValueError("location is out of range")
        if location < 0:
            raise ValueError("location must be positive")
        body_ident = self._get_body_indentation()

        if len(nodes) > 1:
            raise NotImplementedError("only one node can be added at a time")


        self.node.body.insert(location, nodes[0])
        self._add_newlines()
        self._fix_indentation(body_ident)

    def _add_newlines(self):
        for node in self.node.body:
            if getattr(node, 'matcher', None) is None:
                node_source = GetSource(node, assume_no_indent=True)
            else:
                node_source = node.matcher.GetSource()
            if node_source.endswith("\n"):
                continue
            node.matcher.add_newline_to_source()

    def _get_body_indentation(self):
        ident = 0
        for stmt in self.node.body:
            stmt_ident = stmt.col_offset
            if stmt_ident > ident and ident == 0:
                ident = stmt_ident
            elif stmt_ident != ident and ident != 0:
                raise ValueError('illegal ident')
        return ident

    def _fix_indentation(self,  body_ident):
        for node in self.node.body:
            if getattr(node, 'matcher', None) is None:
                raise NotImplementedError("only nodes with matcher are supported")
            else:
                node_matcher = node.matcher
            node_matcher.FixIndentation(body_ident)
