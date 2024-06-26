import ast
import sys

import pytest

from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node.body_manipulator import BodyManipulator
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.manipulate_tests.base_test_utils_manipulate import bcolors

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
            ast.parse(source)
            module_node = GetNodeFromInput(source, get_module=True)

            module_matcher  = GetDynamicMatcher(module_node)
            original_source_after_match = module_matcher.do_match(source)
            assert original_source_after_match == source
            body_manipulator = BodyManipulator(module_node.body)
            body_manipulator.remove_node(node_index)
            source_after_remove = GetSource(module_node)
            assert source_after_remove == expected
            # for index in source_for_remove_tests['inject_to_indexes']:
            #     ast.parse(injected_source)
            #     body_block = module_node.body

    #
    # def _create_nodes(self, capsys, injected_source, original_source, injected_second_source='', is_module=False):
    #     self._capture_source(capsys, original_source, 'original source:', bcolors.OKBLUE)
    #     if_node = self._create_if_node(original_source, is_module)
    #     injected_node, injected_node_source = self._create_injected_node(injected_source, injected_second_source)
    #     return if_node, injected_node

    # def _create_injected_node(self, injected_source, injected_second_source):
    #     injected_node_source = injected_source
    #     #module = False
    #     #if injected_second_source:
    #     injected_node_source +=  injected_second_source
    #     #    module=True
    #     injected_node = GetNodeFromInput(injected_node_source, get_module=True)
    #     injected_node_matcher = GetDynamicMatcher(injected_node)
    #     injected_node_matcher.do_match(injected_node_source)
    #     source_from_matcher = injected_node_matcher.GetSource()
    #     assert source_from_matcher == injected_node_source
    #     source_from_get_source = GetSource(injected_node, assume_no_indent=True)
    #     assert source_from_get_source == injected_node_source
    #     return injected_node, injected_node_source

    # def _create_if_node(self, original_source, is_module):
    #     node = GetNodeFromInput(original_source, get_module=is_module)
    #     node_matcher = GetDynamicMatcher(node)
    #     node_matcher.do_match(original_source)
    #     if_node_source = node_matcher.GetSource()
    #     assert if_node_source == original_source
    #     # if_node.matcher = if_node_matcher
    #     return node

    def _capture_source(self, capsys, source, title, color, ignore_ident=False):
        if not ignore_ident:
            try:
                compile(source, '<string>', mode='exec')
            except Exception as e:
                assert False, f'Error in source compilation'
        print(color + '\n' + title + '\n' + source + bcolors.ENDC)
        out, _ = capsys.readouterr()
        sys.stdout.write(out)

    def _source_after_composition(self, if_node, capsys):
        composed_source = GetSource(if_node, assume_no_indent=True)
        self._capture_source(capsys, composed_source, 'Modified source', bcolors.OKCYAN)
        return composed_source
