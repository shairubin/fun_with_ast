import pytest

from fun_with_ast.manipulate_node import create_node as create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class AssertMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Assert(create_node.Name('a'))
        string = 'assert a\n'
        self._verify_match(node, string)

    def testMatchWithMessage(self):
        node = create_node.Assert(create_node.Name('a'),
                                  create_node.Constant('message', "\""))
        string = 'assert a, "message"\n'
        self._verify_match(node, string)

    def testNoMatchWithMessage(self):
        node = create_node.Assert(create_node.Name('a'),
                                  create_node.Constant('message', "'"))
        string = 'assert a, "message"\n'
        self._verify_match(node,string)
    def testNoMatchWithMessage(self):
        node = create_node.Assert(create_node.Name('a'),
                                  create_node.Constant('message', "'"))
        string = "assert a, 'message'\n"
        self._verify_match(node,string)

    def testNoMatchWithMessage2(self):
        string = "assert a, 'message'\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testNoMatchWithMessage3(self):
        string = """assert isinstance(
        operands, (list, tuple)
    ),  "Cond" """
        node = GetNodeFromInput(string)
        self._verify_match(node, string, trim_suffix_spaces=True)
    def testNoMatchWithMessage3_1(self):
        string = """assert isinstance(
        operands, (list, tuple)
    ),  "Cond" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)
    def testNoMatchWithMessage3_2(self):
        string = """assert isinstance(
        operands, (list, tuple)
    ),  "Cond" """
        node = GetNodeFromInput(string)
        with pytest.raises(AssertionError): # 'Expr node does not support training white spaces'
            self._verify_match(node, string, trim_suffix_spaces=False)

    def testNoMatchWithMessage4(self):
        string = """assert Foo((a, b)), "Cond" """
        node = GetNodeFromInput(string)
        self._verify_match(node, string, trim_suffix_spaces=True)
    def testNoMatchWithMessage4_1(self):
        string = """assert Foo(a, b), "Cond" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)
    def testNoMatchWithMessage4_2(self):
        string = """assert Foo(a, b), "Cond" """
        node = GetNodeFromInput(string, get_module=True)
        with pytest.raises(AssertionError): # 'Module node support trailing  white spaces'
            self._verify_match(node, string, trim_suffix_spaces=True)