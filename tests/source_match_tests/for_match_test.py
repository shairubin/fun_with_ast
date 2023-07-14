import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ForMatcherTest(BaseTestUtils):
    def testSimpleFor(self):
        node = create_node.For(create_node.Name('x'), create_node.Name('y'), [create_node.Pass()])
        string = 'for x in y:\n pass'
        self._verify_match(node, string)

    def testnoMatchSimpleFor(self):
        node = create_node.For(create_node.Name('x'), create_node.Name('y'), [create_node.Pass()])
        string = 'for z in y:\n pass'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testNoMatchSimpleFor2(self):
        node = create_node.For(create_node.Name('x'), create_node.Name('y'), [create_node.Pass()])
        string = 'for x in yy:\n pass'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testNoMatchSimpleFor3(self):
        node = create_node.For(create_node.Name('x'), create_node.Name('y'), [create_node.Pass()])
        string = 'for x in y:\n a=1'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testSimpleFor2(self):
        string = 'for x in y:\n a=1'
        for_node = GetNodeFromInput(string)
        self._verify_match(for_node, string)

    def testSimpleFor3(self):
        string = 'for x in y:\n a=1\n b=2'
        for_node = GetNodeFromInput(string)
        self._verify_match(for_node, string)

    def testSimpleFor4(self):
        string = 'for x in y:\n if a == 1:\n  b=2'
        for_node = GetNodeFromInput(string)
        self._verify_match(for_node, string)

    def testSimpleFor5(self):
        string = 'for x in y:\n if a == 1:\n  b=2\n  c=3'
        for_node = GetNodeFromInput(string)
        self._verify_match(for_node, string)
    def testSimpleFor6(self):
        string = "for x in y:\n if a == 1:\n  b=2\n  print('fun with ast')\n"
        for_node = GetNodeFromInput(string)
        self._verify_match(for_node, string)

    def testSimpleFor7(self):
        string = "for x in y:\n a('fun with ast')\n pass"
        for_node = GetNodeFromInput(string)
        self._verify_match(for_node, string)

    def testSimpleFor8(self):
        string = "for x in range(1,15):\n a('fun with ast')\n pass"
        for_node = GetNodeFromInput(string)
        self._verify_match(for_node, string)

    def testSimpleFor9(self):
        string = "for i in range( 1, x):\n a('fun with ast')\n pass"
        for_node = GetNodeFromInput(string)
        self._verify_match(for_node, string)

    def testSimpleFor10(self):
        string = "for x in range(1,15):\n if (a.b(1)==True):\n  a('fun with ast')\n  pass"
        for_node = GetNodeFromInput(string)
        self._verify_match(for_node, string)
