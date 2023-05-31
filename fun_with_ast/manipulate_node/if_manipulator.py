import ast

from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node.body_manipulator import BodyManipulator

from dataclasses import dataclass

from fun_with_ast.source_matchers.body import BodyPlaceholder


@dataclass
class IfManipulatorConfig():
    body_index: int
    location_in_body_index: int


class ManipulateIfNode():
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
        body_manipulator = BodyManipulator(body_block_to_manipulate)
        body_manipulator.inject_node(node_to_inject, self.config.location_in_body_index)

    def get_body_orelse_source(self):
        if self.config.body_index ==0:
            source = self.node.matcher.body_placeholder.GetSource(self.node)
        elif self.config.body_index ==1:
            source = self.node.matcher.orelse_placeholder.GetSource(self.node)
        else:
            raise ValueError('Illegal body index')
        return source

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
