from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateAnnAssignTest(CreateNodeTestBase):

    def testAnnAssignSimple(self):
        expected_string = 'a:int'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.AnnAssign('a:int')
        self.assertNodesEqual(expected_node, test_node)

    def testAnnAssignSimpleAttribute(self):
        expected_string = 'a.b:int'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.AnnAssign('a.b:int')
        self.assertNodesEqual(expected_node, test_node)
    def testAnnAssignSimpleSubsript(self):
        expected_string = 'a[1tests]:int'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.AnnAssign('a.b:int')
        self.assertNodesEqual(expected_node, test_node)
