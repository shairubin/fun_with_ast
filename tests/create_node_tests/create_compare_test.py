from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateCompareTest(CreateNodeTestBase):

    def testBasicCompare(self):
        expected_string = 'a < b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            create_node.Lt(),
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testMultipleCompare(self):
        expected_string = 'a < b < c'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            create_node.Lt(),
            create_node.Name('b'),
            create_node.Lt(),
            create_node.Name('c'))
        self.assertNodesEqual(expected_node, test_node)

    def testEq(self):
        expected_string = 'a == b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            '==',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testNotEq(self):
        expected_string = 'a != b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            '!=',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testLt(self):
        expected_string = 'a < b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            '<',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testLtE(self):
        expected_string = 'a <= b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            '<=',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testGt(self):
        expected_string = 'a > b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            '>',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testGtE(self):
        expected_string = 'a >= b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            '>=',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testIs(self):
        expected_string = 'a is b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            'is',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testIsNot(self):
        expected_string = 'a is not b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            'is not',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testIn(self):
        expected_string = 'a in b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            'in',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testNotIn(self):
        expected_string = 'a not in b'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Name('a'),
            'not in',
            create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)
    def testFunctionCall(self):
        expected_string = 'a.b() == 1'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Call310('a.b'),
            '==',
            create_node.Num('1'))
        self.assertNodesEqual(expected_node, test_node)

    def testFunctionCall2(self):
        expected_string = '(a.b(2) >= 1)'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Compare(
            create_node.Call310('a.b', args=[create_node.Num('2')]),
            '>=',
            create_node.Num('1'))
        self.assertNodesEqual(expected_node, test_node)
