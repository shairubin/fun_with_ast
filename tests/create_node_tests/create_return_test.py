from manipulate_node import create_node
from manipulate_node.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateReturnTest(CreateNodeTestBase):

    def testRetrunSigleValueInt(self):
        expected_string = 'return 1'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Return(1)
        self.assertNodesEqual(expected_node, test_node)

    def testRetrunSigleValueStr(self):
        expected_string = "return '1'"
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Return('1')
        self.assertNodesEqual(expected_node, test_node)

    def testRetrunSigleValueName(self):
        expected_string = "return a"
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Return(create_node.Name('a'))
        self.assertNodesEqual(expected_node, test_node)
