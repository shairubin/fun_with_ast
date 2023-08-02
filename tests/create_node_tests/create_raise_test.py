import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateRaiseTest(CreateNodeTestBase):


    def testCreateRaiseSimple(self):
        expected_string = 'raise'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Raise()
        self.assertNodesEqual(expected_node, test_node)
    def testCreateRaiseSimple1(self):
        expected_string = 'raise e'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Raise(exception=create_node.Name('e'))
        self.assertNodesEqual(expected_node, test_node)

    def testCreateRaiseSimple2(self):
        expected_string = 'raise e from d'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Raise(exception=create_node.Name('e'), cause=create_node.Name('d'))
        self.assertNodesEqual(expected_node, test_node)

    def testCreateRaiseNotEqual(self):
        expected_string = 'raise e from c'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Raise(exception=create_node.Name('e'), cause=create_node.Name('d'))
        with pytest.raises(AssertionError):
            self.assertNodesEqual(expected_node, test_node)
