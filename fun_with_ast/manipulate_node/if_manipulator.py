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
        if not isinstance(node, ast.If):
            raise ValueError('node must be ast.If')
        self.is_elif = True if hasattr(node, 'is_elif') and node.is_elif else False


    def add_nodes(self, nodes: list, location: IfManipulatorConfig):
        self._validate_rules_for_insertion(location, nodes)
        body_block_to_manipulate = self._get_block(location.body_index)
        ident = self._get_node_indentation(body_block_to_manipulate)
        node_to_inject = nodes[0]
        source_to_inject = GetSource(node_to_inject, assume_no_indent=True)
        node_to_inject.matcher.FixIndentation(ident)

        body_block_to_manipulate.insert(location.location_in_body_index, node_to_inject)
        self._add_newlines(body_block_to_manipulate)

    def get_body_source(self, IFManipulatorConfig):
        body_block = self._get_block(IFManipulatorConfig.body_index)
        return GetSource(body_block)

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
        if location.body_index < 0:
            raise ValueError('Illegal body index')
        if location.body_index > 1:
            raise NotImplementedError('elif not supported yet')
        if location.body_index == 1:
            if self.is_elif:
                if not isinstance(self.node.orelse[0], ast.If):
                    raise ValueError('orelse is not ast.If and is_elif is True')
                if len(self.node.orelse) > 1:
                    raise ValueError('orelse length is larger than 1 and is_elif is True')
        if location.body_index == 1 and not self.node.orelse:
            raise ValueError('No oresle in If but index body is 1')

    def _add_newlines(self, body_block):
        for node in body_block:
            node_source = node.matcher.GetSource()
            if node_source.endswith("\n"):
                continue
            node.matcher.add_newline_to_source()

    def _get_node_indentation(self, node):
        ident = 0
        for stmt in node:
            stmt_ident = stmt.col_offset
            if stmt_ident > ident and ident == 0:
                ident = stmt_ident
            elif stmt_ident != ident and ident != 0:
                raise ValueError('illegal ident')
        return ident

    def _get_block(self, body_index):
        while True:
            if body_index < 0:
                raise ValueError('Illegal body index')
            if body_index == 0:
                return self.node.body
            if body_index == 1 and not self.node.orelse:
                ValueError('No oresle in If')
            if body_index == 1:
                if self.is_elif:
                    return self.node.orelse[0].body
                return self.node.orelse
            body_index -= 2
