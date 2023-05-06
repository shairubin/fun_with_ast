import create_node
from fun_with_ast.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateBinOpTest(CreateNodeTestBase):

    def testBinOpWithAdd(self):
        expected_string = 'a + b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '+',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBinOpWithSub(self):
        expected_string = 'a - b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '-',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBinOpWithMult(self):
        expected_string = 'a * b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '*',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBinOpWithDiv(self):
        expected_string = 'a / b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '/',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBinOpWithFloorDiv(self):
        expected_string = 'a // b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '//',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBinOpWithMod(self):
        expected_string = 'a % b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '%',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBinOpWithPow(self):
        expected_string = 'a ** b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '**',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBinOpWithLShift(self):
        expected_string = 'a << b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '<<',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBinOpWithRShift(self):
        expected_string = 'a >> b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '>>',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBinOpWithBitOr(self):
        expected_string = 'a | b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '|',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBinOpWithBitXor(self):
        expected_string = 'a ^ b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '^',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testBinOpWithBitAnd(self):
        expected_string = 'a & b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.BinOp(
            create_node.Name('a'),
            '&',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)
