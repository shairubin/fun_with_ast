from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateBoolOpTest(CreateNodeTestBase):

    def testBoolOpWithAnd(self):
        expected_string = 'a and b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BoolOp(
            create_node.Name('a'),
            'and',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBoolOpWithOr(self):
        expected_string = 'a or b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BoolOp(
            create_node.Name('a'),
            'or',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBoolOpWithAndOr(self):
        expected_string = 'a and b or c'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BoolOp(
            create_node.Name('a'),
            'and',
            create_node.Name('b'),
            'or',
            create_node.Name('c'))
        self.assertNodesEqual(expected_node, test_node)

    def testBoolOpWithOrAnd(self):
        expected_string = 'a or b and c'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BoolOp(
            create_node.Name('a'),
            'or',
            create_node.Name('b'),
            'and',
            create_node.Name('c'))
        self.assertNodesEqual(expected_node, test_node)

    def testBoolOpWithOrAnd2(self):
        expected_string = '(a or b) and c'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BoolOp(
            create_node.BoolOp(create_node.Name('a'), 'or', create_node.Name('b')),
            'and',
            create_node.Name('c'))
        self.assertNodesEqual(expected_node, test_node)
