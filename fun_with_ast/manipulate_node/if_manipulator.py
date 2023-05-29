import ast

from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node.body_manipulator import BodyManipulator

from dataclasses import dataclass
@dataclass
class IfManipulatorConfig():
    body_index: int
    location_in_body_index: int


class ManipulateIfNode(BodyManipulator):
    def __init__(self, node, config: IfManipulatorConfig):
        self.node = node
        self.config = config
        if not isinstance(node, ast.If):
            raise ValueError('node must be ast.If')
        self.is_elif = True if hasattr(node, 'is_elif') and node.is_elif else False


    def add_nodes(self, nodes: list):
        self._validate_rules_for_insertion(nodes)
        node_to_inject = nodes[0]
        body_block_to_manipulate = self._get_block(self.config.body_index)
        self.inject_to_body(body_block_to_manipulate, node_to_inject, self.config.location_in_body_index)

    # def get_body_source(self, IFManipulatorConfig):
    #     body_block = self._get_block(IFManipulatorConfig.body_index)
    #     return GetSource(body_block)

    # def _handle_expr_node(self, nodes):
    #     expr_node = nodes[0]
    #     module_node = create_node.Module(expr_node)
    #     expr_value_source = expr_node.value.matcher.GetSource()
    #     if expr_value_source.endswith("\n"):
    #         raise NotImplementedError("expr value source cannot end with newline")
    #     else:
    #         expr_value_source += "\n"
    #     placeholder = BodyPlaceholder('body')
    #     placeholder.Match(module_node, expr_value_source)
    #     return module_node

    def _validate_rules_for_insertion(self, nodes):
        if len(nodes) > 1:
            raise NotImplementedError("only one node can be added at a time")
        if self.config.location_in_body_index > len(self.node.body):
            raise ValueError("location is out of range")
        if self.config.location_in_body_index < 0:
            raise ValueError("location must be positive")
        if self.config.body_index < 0:
            raise ValueError('Illegal body index')
        if self.config.body_index > 1:
            raise NotImplementedError('elif not supported yet')
        if self.config.body_index == 1:
            if self.is_elif:
                if not isinstance(self.node.orelse[0], ast.If):
                    raise ValueError('orelse is not ast.If and is_elif is True')
                if len(self.node.orelse) > 1:
                    raise ValueError('orelse length is larger than 1 and is_elif is True')
        if self.config.body_index == 1 and not self.node.orelse:
            raise ValueError('No oresle in If but index body is 1')

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
