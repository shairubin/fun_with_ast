from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateListTest(CreateNodeTestBase):

    def testListLoad(self):
        expected_string = 'a = ["b"]'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.List(
            create_node.Str('b'), ctx_type=create_node.CtxEnum.LOAD)
        self.assertNodesEqual(expected_node, test_node)

    def testListStore(self):
        expected_string = '[a, b] = ["c", "d"]'
        expected_node = GetNodeFromInput(expected_string).targets[0]
        test_node = create_node.List(
            create_node.Name('a'),
            create_node.Name('b'),
            ctx_type=create_node.CtxEnum.STORE)
        self.assertNodesEqual(expected_node, test_node)

    def testDeleteInvalid(self):
        expected_string = 'del [a, b]'
        expected_node = GetNodeFromInput(expected_string).targets[0]
        test_node = create_node.List(
            create_node.Name('a'),
            create_node.Name('b'),
            ctx_type=create_node.CtxEnum.DEL)
        self.assertNodesEqual(expected_node, test_node)

    def testListOverridesInnerCtx(self):
        expected_string = 'a = [b, c]'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.List(
            create_node.Name('b', ctx_type=create_node.CtxEnum.DEL),
            create_node.Name('c', ctx_type=create_node.CtxEnum.STORE),
            ctx_type=create_node.CtxEnum.LOAD)
        self.assertNodesEqual(expected_node, test_node)
