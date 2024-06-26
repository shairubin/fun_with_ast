import ast
import sys

import pytest

from fun_with_ast.get_source import GetSource
from tests.manipulate_tests.base_test_utils_manipulate import bcolors

input_legend = ('inject-source', 'location', 'original-if', 'expected', 'match-expected', 'injected_second_source')


module_1 = """class foo():
  def __init__(self):
    a=1
    def bar(self):
        b=2
    c=4"""
module_2 = """class foo():
    def bar():
        with a as p:
            a=1
            b=2
            c=3"""
@pytest.fixture(params=[
    ({"source": module_1, "injected_source": "d=10\n",
      "inject_into_body":"module_node.body" , "inject_to_indexes": [(0,0,0),(1,6,0)]}),
    ({"source": module_1, "injected_source": "d=11\n",
      "inject_into_body": "module_node.body[0].body", "inject_to_indexes": [(0, 1,2), (1,6,2)]}),
    ({"source": module_1, "injected_source": "d=12\n",
      "inject_into_body": "module_node.body[0].body[0].body",
      "inject_to_indexes": [(0, 2, 4), (1, 3, 4), (2,5,4), (3,6,4)]}),
    ({"source": module_1, "injected_source": "d=13\n",
      "inject_into_body": "module_node.body[0].body[0].body[1].body",
      "inject_to_indexes": [(0, 4, 8), (1,5,8)]}),
    ({"source": module_2, "injected_source": "d=14\n",
      "inject_into_body": "module_node.body[0].body[0].body",
      "inject_to_indexes": [(0, 2, 8), (1, 6, 8)]}),
    ({"source": module_2, "injected_source": "d=14\n",
      "inject_into_body": "module_node.body[0].body[0].body",
      "inject_to_indexes": [(0, 2, 8), (1, 6, 8)]}),
    ({"source": module_2, "injected_source": "d=15\n",
      "inject_into_body": "module_node.body[0].body[0].body[0].body",
      "inject_to_indexes": [(0, 3, 12), (1, 4, 12), (2, 5, 12) , (3, 6, 12)]}),
    ({"source": module_2, "injected_source": "d=16\n",
      "inject_into_body": "module_node.body[0].body",
      "inject_to_indexes": [(0, 1, 4), (1, 6, 4)]}),

])
def source_for_remove_tests(request):
    yield request.param

class TestRemoveNodeManipulation:

    def test_dynamic_Module_Body_Remove_Manipulation(self, source_for_remove_tests, capsys):
            source = source_for_remove_tests['source']
            ast.parse(source)
            injected_source = source_for_remove_tests['injected_source']
            for index in source_for_remove_tests['inject_to_indexes']:
                ast.parse(injected_source)
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
