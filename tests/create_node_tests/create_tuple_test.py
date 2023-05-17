from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateTupleTest(CreateNodeTestBase):

    def testTupleLoad(self):
        expected_string = 'a = ("b",)'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Tuple([create_node.Constant('b')], ctx_type=create_node.CtxEnum.LOAD)
        self.assertNodesEqual(expected_node, test_node)

    def testTupleWithStrings(self):
        expected_string = 'a = (b,c)'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Tuple(['b', 'c'],
                                      ctx_type=create_node.CtxEnum.LOAD)
        self.assertNodesEqual(expected_node, test_node)

    def testTupleStore(self):
        expected_string = '(a, b) = ["c", "d"]'
        expected_node = GetNodeFromInput(expected_string).targets[0]
        test_node = create_node.Tuple(['a', 'b'],
                                      ctx_type=create_node.CtxEnum.STORE)
        self.assertNodesEqual(expected_node, test_node)

    def testDeleteInvalid(self):
        expected_string = 'del (a, b)'
        expected_node = GetNodeFromInput(expected_string).targets[0]
        test_node = create_node.Tuple(['a', 'b'],
                                      ctx_type=create_node.CtxEnum.DEL)
        self.assertNodesEqual(expected_node, test_node)

    def testTupleOverridesInnerCtx(self):
        expected_string = 'a = (b, c)'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Tuple(['b', 'c'],
                                      ctx_type=create_node.CtxEnum.LOAD)
        self.assertNodesEqual(expected_node, test_node)
