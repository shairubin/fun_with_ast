import create_node
from fun_with_ast.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateDictComprehensionTest(CreateNodeTestBase):

    def testBasicDictComprehension(self):
        expected_string = '{a: b for c in d}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.DictComp('a', 'b', 'c', 'd')
        self.assertNodesEqual(expected_node, test_node)

    def testBasicDictComprehensionWithIfs(self):
        expected_string = '{a: b for c in d if e < f}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.DictComp(
            'a', 'b', 'c', 'd',
            create_node.Compare('e', '<', 'f'))
        self.assertNodesEqual(expected_node, test_node)

class CreateListComprehensionTest(CreateNodeTestBase):

    def testBasicListComprehension(self):
        expected_string = '[a for a in b]'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.ListComp('a', 'a', 'b')
        self.assertNodesEqual(expected_node, test_node)

    def testBasicListComprehensionWithIfs(self):
        expected_string = '[a for a in b if c < d]'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.ListComp(
            'a', 'a', 'b',
            create_node.Compare('c', '<', 'd'))
        self.assertNodesEqual(expected_node, test_node)

class CreateSetComprehensionTest(CreateNodeTestBase):

    def testBasicSetComprehension(self):
        expected_string = '{a for a in b}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.SetComp('a', 'a', 'b')
        self.assertNodesEqual(expected_node, test_node)

    def testBasicSetComprehensionWithIfs(self):
        expected_string = '{a for a in b if c < d}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.SetComp(
            'a', 'a', 'b',
            create_node.Compare('c', '<', 'd'))
        self.assertNodesEqual(expected_node, test_node)
