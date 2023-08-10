from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateImportFromTest(CreateNodeTestBase):

    def testImportFrom(self):
        expected_string = """from x import y"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.ImportFrom('x', ['y'])
        self.assertNodesEqual(expected_node, test_node)

    def testNoMatchImportFrom(self):
        expected_string = """from x import y"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.ImportFrom('y', ['x'])
        self.assertNodesEqual(expected_node, test_node, equal=False)

    def testImportFrom2(self):
        expected_string = """from x import y,z """
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.ImportFrom('x', ['y', 'z'])
        self.assertNodesEqual(expected_node, test_node)

    def testImportFrom3(self):
        expected_string = """from x import (y,
        z) """
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.ImportFrom('x', ['y', 'z'])
        self.assertNodesEqual(expected_node, test_node)

    def testImportFrom4(self):
        expected_string = """from . import (y,
        z) """
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.ImportFrom(None, ['y', 'z'], level=1)
        self.assertNodesEqual(expected_node, test_node)
