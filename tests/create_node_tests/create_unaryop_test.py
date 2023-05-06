import create_node
from fun_with_ast.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateUnaryOpTest(CreateNodeTestBase):

    def testUnaryOpWithPositive(self):
        expected_string = '+b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.UnaryOp(
            '+',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testUnaryOpWithSub(self):
        expected_string = '-b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.UnaryOp(
            '-',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testUnaryOpWithNot(self):
        expected_string = 'not b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.UnaryOp(
            'not',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testUnaryOpWithInvert(self):
        expected_string = '~b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.UnaryOp(
            '~',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)
