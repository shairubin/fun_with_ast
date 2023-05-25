import unittest
import pytest

from fun_with_ast.get_source import GetSource
from fun_with_ast.dynamic_matcher import GetDynamicMatcher
from fun_with_ast.manipulate_node.create_node import GetNodeFromInput
from fun_with_ast.manipulate_node.if_manipulator import ManipulateIfNode


@pytest.fixture(params=['a.b()\n', \
                        'a.c()\n', \
                        'a=44'
                        ])
def injected_source(request):
    yield request.param
class TestIfManupulation:


    def test_If_Manipulation(self, injected_source):
        original_if_source = 'if (c.d()):\n   a=1'
        if_node = self._create_if_node(original_if_source)
        injected_node, injected_node_source = self._create_injected_node(injected_source)
        manipulator = ManipulateIfNode(if_node)
        manipulator.add_nodes_to_body([injected_node],1)
        composed_source = GetSource(if_node, assume_no_indent=True)
        add_new_line = '' if injected_source.endswith('\n') else '\n'
        expected_source = original_if_source + '\n   ' + injected_node_source + add_new_line
        assert expected_source == composed_source


    def test_If_Else_Manipulation(self, injected_source):
        original_if_source = 'if (c.d()):\n   a=1\nelse:\n   b=2'
        if_node = self._create_if_node(original_if_source)
        injected_node, injected_node_source = self._create_injected_node(injected_source)
        manipulator = ManipulateIfNode(if_node)
        manipulator.add_nodes_to_body([injected_node],1)
        composed_source = GetSource(if_node, assume_no_indent=True)
        add_new_line = '\n' if not injected_source.endswith('\n') else ''
        expected_source = original_if_source.replace('a=1\n', 'a=1\n   '+injected_source + add_new_line )
        assert expected_source == composed_source

    def _create_injected_node(self, injected_source):
        injected_node_source = injected_source
        injected_node = GetNodeFromInput(injected_node_source)
        injected_node_matcher = GetDynamicMatcher(injected_node)
        injected_node_matcher.Match(injected_node_source)
        source_from_matcher = injected_node_matcher.GetSource()
        assert source_from_matcher == injected_node_source
        source_from_get_source = GetSource(injected_node, assume_no_indent=True)
        assert source_from_get_source == injected_node_source
        return injected_node, injected_node_source

    def _create_if_node(self, original_if_source):
        if_node = GetNodeFromInput(original_if_source)
        if_node_matcher = GetDynamicMatcher(if_node)
        if_node_matcher.Match(original_if_source)
        if_node_source = if_node_matcher.GetSource()
        assert if_node_source == original_if_source
        return if_node
