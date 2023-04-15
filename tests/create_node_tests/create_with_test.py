import create_node
from create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateWithTest(CreateNodeTestBase):

    def testBasicWith(self):
        expected_string = 'with a:\n  pass\n'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.With(
            [create_node.withitem('a')], [(create_node.Pass())])
        self.assertNodesEqual(expected_node, test_node)

    def testBasicWithAs(self):
        expected_string = 'with a as b:\n  pass\n'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.With([create_node.withitem('a', optional_vars='b')], [create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testWithAsTuple(self):
        expected_string = 'with a as (b, c):\n  pass\n'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.With(
            [create_node.withitem('a', create_node.Tuple(['b', 'c'], ctx_type=create_node.CtxEnum.STORE))],
            [create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)
