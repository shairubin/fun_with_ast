import create_node
from create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateAssignTest(CreateNodeTestBase):

    def testAssignWithSimpleString(self):
        expected_string = 'a = "b"'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Assign('a', create_node.Str('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testAssignListWithSimpleString(self):
        expected_string = 'a=c="b"'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Assign(['a', 'c'], create_node.Str('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testAssignWithNode(self):
        expected_string = 'a = "b"'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Assign(
            create_node.Name('a', ctx_type=create_node.CtxEnum.STORE),
            create_node.Str('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testAssignWithTuple(self):
        expected_string = '(a, c) = "b"'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Assign(
            create_node.Tuple(['a', 'c'], ctx_type=create_node.CtxEnum.STORE),
            create_node.Str('b'))
        self.assertNodesEqual(expected_node, test_node)
