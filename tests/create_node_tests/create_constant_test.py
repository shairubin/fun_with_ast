from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase

class CreateStrTest(CreateNodeTestBase):

    def testStr(self):
        expected_string = '"a"'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Str('a')
        self.assertNodesEqual(expected_node, test_node)


class CreateNumTest(CreateNodeTestBase):

    def testNumWithInteger(self):
        expected_string = '15'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Num('15')
        self.assertNodesEqual(expected_node, test_node)

    def testNumWithHex(self):
        expected_string = '0xa5'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Num('0xa5')
        self.assertNodesEqual(expected_node, test_node)

    def testNumWithFloat(self):
        expected_string = '0.25'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Constant(0.25)
        self.assertNodesEqual(expected_node, test_node)
