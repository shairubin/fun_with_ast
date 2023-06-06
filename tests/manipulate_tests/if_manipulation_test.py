import sys
import pytest

from fun_with_ast.get_source import GetSource
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.manipulate_node.create_node import GetNodeFromInput
from fun_with_ast.manipulate_node.if_manipulator import ManipulateIfNode, IfManipulatorConfig

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@pytest.fixture(params=['a.b()\n', \
                        'a.c()\n', \
                        'a=44',
                        "s='fun_with_ast'"
                        ])
def injected_source(request):
    yield request.param

@pytest.fixture(params=[#('   pass\n', '   a=1\n', 0, 'c.d():'),
                        ('   a=1\n', '   pass', 0, 'c.d(): # comment'),
                        ("  if x%2 == 0:\n    print(\"x is a positive even number\")\n  else:\n    print(\"x is a positive odd number\")\n",
                         '  a=1', 0, 'a>2: #comment'),
                        ('#line comment \n   pass #comment 1\n', '   a=1 #comment 2', 0, 'a and not b and not not c:' ),
                        ("  if x%2 == 0:\n    print(\"x is a positive even number\")\n  else:\n    print(\"x is a positive odd number\")\n",'  a=1', 1,
                         '(a and not b) or not (not c):'),
                        ('#line comment \n   pass #comment 1\n', '   a=1 #comment 2', 1, 'a+b > c/d+c:'),
                        ('#line comment \n   pass #comment 1\n', '   a=1 #comment 2', 0, 'a+b > c/(d+c):')

                        ])
def body_and_orelse(request):
    yield request.param
class TestIfManupulation:
    def test_If_Manipulation(self, injected_source, capsys):
        original_if_source = 'if (c.d()):\n   a=1'
        if_node, injected_node = self._create_nodes(capsys, injected_source, original_if_source)
        manipulator = ManipulateIfNode(if_node, IfManipulatorConfig(body_index=0, location_in_body_index=0))
        manipulator.add_nodes([injected_node])
        composed_source = self._source_after_composition(if_node, capsys)
        #self._capture_source(capsys, composed_source, 'modified source:', bcolors.OKCYAN)
        add_new_line = '' if injected_source.endswith('\n') else '\n'
        expected_source = original_if_source.replace('   a=1',  '   '+injected_source + add_new_line +'   a=1\n')
        assert expected_source == composed_source


    def test_If_Else_Manipulation(self, injected_source, capsys):
        original_if_source = 'if ( c.d() ):\n   a=1\nelse:\n   b=2'
        if_node, injected_node = self._create_nodes(capsys, injected_source, original_if_source)
        manipulator = ManipulateIfNode(if_node,IfManipulatorConfig(body_index=1, location_in_body_index=1))
        manipulator.add_nodes([injected_node])
        composed_source = self._source_after_composition(if_node, capsys)
        add_new_line = '\n' if not injected_source.endswith('\n') else ''
        expected_source = original_if_source.replace('b=2', 'b=2\n   '+injected_source + add_new_line )
        assert composed_source == expected_source

    def test_If_elif_Manipulation(self, injected_source, capsys):
        original_if_source = 'if ( c.d() ):\n   a=1\nelif e==2:\n   b=2'
        if_node, injected_node = self._create_nodes(capsys, injected_source, original_if_source)
        manipulator = ManipulateIfNode(if_node, IfManipulatorConfig(body_index=1, location_in_body_index=1))
        manipulator.add_nodes([injected_node])
        composed_source = self._source_after_composition(if_node, capsys)
        add_new_line = '\n' if not injected_source.endswith('\n') else ''
        expected_source = original_if_source.replace('b=2', 'b=2\n   '+injected_source + add_new_line )
        assert composed_source == expected_source

    def test_get_source_body_Manipulation(self, body_and_orelse, capsys):
        body = body_and_orelse[0]
        orelse = body_and_orelse[1]
        body_index = body_and_orelse[2]
        test = body_and_orelse[3]
        original_if_source = 'if '+ test + '\n'   + body + 'else:\n' + orelse
        if_node, injected_node = self._create_nodes(capsys, 'pass', original_if_source)
        manipulator = ManipulateIfNode(if_node, IfManipulatorConfig(body_index=body_index, location_in_body_index=1))
        the_source = manipulator.get_body_orelse_source()
        title = 'Body source:' if body_index == 0 else 'Else source:'
        title = 'test_get_source. ' + title
        self._capture_source(capsys, the_source, title, bcolors.OKGREEN, True)
        if body_index == 0:
            assert the_source == body
        elif body_index == 1:
            assert the_source == orelse
        else:
            raise ValueError("body index can be only 0 or 1")

    def _create_nodes(self, capsys, injected_source, original_if_source):
        self._capture_source(capsys, original_if_source, 'original source:', bcolors.OKBLUE)
        if_node = self._create_if_node(original_if_source)
        injected_node, injected_node_source = self._create_injected_node(injected_source)
        return if_node, injected_node

    def _create_injected_node(self, injected_source):
        injected_node_source = injected_source
        injected_node = GetNodeFromInput(injected_node_source)
        injected_node_matcher = GetDynamicMatcher(injected_node)
        injected_node_matcher._match(injected_node_source)
        source_from_matcher = injected_node_matcher.GetSource()
        assert source_from_matcher == injected_node_source
        source_from_get_source = GetSource(injected_node, assume_no_indent=True)
        assert source_from_get_source == injected_node_source
        return injected_node, injected_node_source

    def _create_if_node(self, original_if_source):
        if_node = GetNodeFromInput(original_if_source)
        if_node_matcher = GetDynamicMatcher(if_node)
        if_node_matcher._match(original_if_source)
        if_node_source = if_node_matcher.GetSource()
        assert if_node_source == original_if_source
        if_node.matcher = if_node_matcher
        return if_node

    def _capture_source(self, capsys, source, title , color, ignore_ident=False):
        if not ignore_ident:
            compile(source, '<string>', mode='exec')
        print(color + '\n' + title + '\n' + source + bcolors.ENDC)
        out, _ = capsys.readouterr()
        sys.stdout.write(out)
    def _source_after_composition(self, if_node, capsys):
        composed_source = GetSource(if_node, assume_no_indent=True)
        self._capture_source(capsys, composed_source, 'Modified source', bcolors.OKCYAN)
        return composed_source
