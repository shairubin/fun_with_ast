import ast

import pytest

from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node.body_manipulator import BodyManipulator
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher

input_legend = ('inject-source', 'location', 'original-if', 'expected', 'match-expected', 'injected_second_source')


module_1 = """
import module
a=1
b=2
c = a-b
"""
@pytest.fixture(params=[
    ({"source": module_1, "removed_source": "import module\n", 'node_index':1,
     "expected":module_1.replace("import module\n", "")}),
    ({"source": module_1, "removed_source": "a=1\n", 'node_index': 2,
      "expected": module_1.replace("a=1\n", "")}),
    ({"source": module_1, "removed_source": "b=2\n", 'node_index': 3,
      "expected": module_1.replace("b=2\n", "")}),
    ({"source": module_1, "removed_source": "c = a-b\n", 'node_index': 4,
      "expected": module_1.replace("c = a-b\n", "")}),

])
def source_for_remove_tests(request):
    yield request.param

class TestRemoveNodeManipulation:

    def test_Module_Body_Remove_Manipulation(self, source_for_remove_tests, capsys):
            source = source_for_remove_tests['source']
            expected = source_for_remove_tests['expected']
            node_index = source_for_remove_tests['node_index']
            removed_source = source_for_remove_tests['removed_source']
            ast.parse(source)
            module_node = GetNodeFromInput(source, get_module=True)

            module_matcher  = GetDynamicMatcher(module_node)
            original_source_after_match = module_matcher.do_match(source)
            assert original_source_after_match == source
            body_manipulator = BodyManipulator(module_node.body)
            body_manipulator.remove_node(node_index, removed_source)
            source_after_remove = GetSource(module_node)
            assert source_after_remove == expected
