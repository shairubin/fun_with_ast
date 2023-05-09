from manipulate_node import create_node
from manipulate_node.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateArgumentsTest(CreateNodeTestBase):

    def testBasic(self):
        expected_string = "f'Fun With Ast'"
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.JoinedStr([create_node.Str('Fun With Ast')])
        self.assertNodesEqual(expected_node, test_node)

    def testBasic2(self):
        string = 'f\"Fun With Ast\"'
        expected_node = GetNodeFromInput(string).value
        test_node = create_node.JoinedStr([create_node.Str('Fun With Ast')])
        self.assertNodesEqual(expected_node, test_node)

    def testPlaceholder(self):
        string = 'f\"Fun {With} Ast\"'
        expected_node = GetNodeFromInput(string).value
        test_node = create_node.JoinedStr([create_node.Str('Fun '), create_node.FormattedValue(create_node.Name('With')), create_node.Str(' Ast')])
        self.assertNodesEqual(expected_node, test_node)
