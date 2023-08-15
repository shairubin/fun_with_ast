import pytest

import fun_with_ast.manipulate_node.get_node_from_input
from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateArgumentsTest(CreateNodeTestBase):

    def testEmpty(self):
        expected_string = """def testFunc():
  pass"""
        expected_node = GetNodeFromInput(expected_string).args
        test_node = create_node.arguments()
        self.assertNodesEqual(expected_node, test_node)

    def testArgs(self):
        expected_string = """def testFunc(a, b):
    pass"""
        expected_node = GetNodeFromInput(expected_string).args
        test_node = create_node.arguments(args=['a', 'b'])
        self.assertNodesEqual(expected_node, test_node)

    def testArgsWithAnnotations(self):
        expected_string = """def testFunc(a:int, b:str):
    pass"""
        expected_node = GetNodeFromInput(expected_string).args
        test_node = create_node.arguments(args=['a:int', 'b:str'])
        self.assertNodesEqual(expected_node, test_node)

    def testArgsWithAnnotations2(self):
        expected_string = """def testFunc(a:int, b:int):
    pass"""
        expected_node = GetNodeFromInput(expected_string).args
        test_node = create_node.arguments(args=['a:int', 'b:str'])
        with pytest.raises(AssertionError):
            self.assertNodesEqual(expected_node, test_node)

    def testStringKwargs(self):
        expected_string = """def testFunc(a='b', c='d'):
  pass"""
        expected_node = GetNodeFromInput(expected_string).args
        test_node = create_node.arguments(
            args=['a', 'c'],
            defaults=[create_node.Str('b'), create_node.Str('d')])
        self.assertNodesEqual(expected_node, test_node)

    def testNameKwargs(self):
        expected_string = """def testFunc(a=b, c=d):
  pass"""
        expected_node = GetNodeFromInput(expected_string).args
        test_node = create_node.arguments(
            args=['a', 'c'],
            defaults=['b', 'd'])
        self.assertNodesEqual(expected_node, test_node)

    def testVararg(self):
        expected_string = """def testFunc(*args):
  pass"""
        expected_node = GetNodeFromInput(expected_string).args
        test_node = create_node.arguments(vararg='args')
        self.assertNodesEqual(expected_node, test_node)

    def testFunctionDefWithKwarg(self):
        expected_string = """def testFunc(**kwargs):
  pass"""
        expected_node = GetNodeFromInput(expected_string).args
        test_node = create_node.arguments(kwarg='kwargs')
        self.assertNodesEqual(expected_node, test_node)
