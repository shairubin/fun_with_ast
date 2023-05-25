import ast
import dataclasses

from fun_with_ast.get_source import GetSource

from fun_with_ast.source_matchers.body import BodyPlaceholder

from fun_with_ast.manipulate_node import create_node
from dataclasses import dataclass
@dataclass
class IfManipulatorConfig():
    body_index: int
    location_in_body_index: int

class ManipulateIfNode():
    def __init__(self, node):
        self.node = node

    def add_nodes_to_body(self, nodes: list, location: IfManipulatorConfig):
        self._validate_rules_for_insertion(location, nodes)
        body_ident = self._get_body_indentation()
        node_to_inject = nodes[0]
        source_to_inject = GetSource(node_to_inject, assume_no_indent=True)
        node_to_inject.matcher.FixIndentation(body_ident)

        if isinstance(node_to_inject, ast.Expr) and False:
           module_node = self._handle_expr_node(nodes)
           self.node.body.insert(location, module_node.body[0])
        else:
            self.node.body.insert(location.location_in_body_index, node_to_inject)
        self._add_newlines()


    def _handle_expr_node(self, nodes):
        expr_node = nodes[0]
        module_node = create_node.Module(expr_node)
        expr_value_source = expr_node.value.matcher.GetSource()
        if expr_value_source.endswith("\n"):
            raise NotImplementedError("expr value source cannot end with newline")
        else:
            expr_value_source += "\n"
        placeholder = BodyPlaceholder('body')
        placeholder.Match(module_node, expr_value_source)
        return module_node

    def _validate_rules_for_insertion(self, location, nodes):
        if len(nodes) > 1:
            raise NotImplementedError("only one node can be added at a time")
        if location.location_in_body_index > len(self.node.body):
            raise ValueError("location is out of range")
        if location.location_in_body_index < 0:
            raise ValueError("location must be positive")
        if location.body_index != 0:
            raise NotImplementedError
    def _add_newlines(self):
        for node in self.node.body:
            if isinstance(node, ast.Expr) :
                node_source = node.matcher.GetSource()
                node = node.value
#            elif isinstance(node, ast.stmt) and not node.parent:
#                raise NotImplementedError("stmts are not supported")
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

    # def _fix_indentation(self,  body_ident):
    #     for node in self.node.body:
    #         if isinstance(node , ast.Expr):
    #             node.value.matcher.FixIndentation(body_ident)
    #         else:
    #             node.matcher.FixIndentation(body_ident)
