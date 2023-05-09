from manipulate_node import create_node
from manipulate_node.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class VarReferenceTest(CreateNodeTestBase):

    def testNoDotSeparated(self):
        expected_string = 'b = a'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.VarReference(
            'a', ctx_type=create_node.CtxEnum.LOAD)
        self.assertNodesEqual(expected_node, test_node)

    def testSingleDotSeparated(self):
        expected_string = 'b = a.c'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.VarReference(
            'a', 'c',
            ctx_type=create_node.CtxEnum.LOAD)
        self.assertNodesEqual(expected_node, test_node)

    def testDoubleDotSeparated(self):
        expected_string = 'b = a.c.d'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.VarReference(
            'a', 'c', 'd',
            ctx_type=create_node.CtxEnum.LOAD)
        self.assertNodesEqual(expected_node, test_node)

    def testStringAsFirst(self):
        expected_string = 'b = "a".c.d'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.VarReference(
            create_node.Str('a'), 'c', 'd',
            ctx_type=create_node.CtxEnum.LOAD)
        self.assertNodesEqual(expected_node, test_node)
