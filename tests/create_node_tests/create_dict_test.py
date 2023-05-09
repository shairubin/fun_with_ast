from manipulate_node import create_node
from manipulate_node.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateDictTest(CreateNodeTestBase):

    def testDictWithStringKeys(self):
        expected_string = '{"a": "b"}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Dict(
            [create_node.Str('a')],
            [create_node.Str('b')])
        self.assertNodesEqual(expected_node, test_node)

    def testDictWithNonStringKeys(self):
        expected_string = '{a: 1}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Dict(
            [create_node.Name('a')],
            [create_node.Num(1)])
        self.assertNodesEqual(expected_node, test_node)

    def testDictWithNoKeysOrVals(self):
        expected_string = '{}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Dict([], [])
        self.assertNodesEqual(expected_node, test_node)

    def testDictRaisesErrorIfNotMatchingLengths(self):
        with self.assertRaises(ValueError):
            unused_test_node = create_node.Dict(
                [create_node.Str('a')],
                [])
