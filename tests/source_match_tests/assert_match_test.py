from fun_with_ast.manipulate_node import create_node as create_node
from tests.source_match_tests.base_test_utils import BaseTestUtils


class AssertMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Assert(create_node.Name('a'))
        string = 'assert a\n'
        self._verify_match(node, string)

    def testMatchWithMessage(self):
        node = create_node.Assert(create_node.Name('a'),
                                  create_node.Constant('message', "\""))
        string = 'assert a, "message"\n'
        self._verify_match(node, string)

    def testNoMatchWithMessage(self):
        node = create_node.Assert(create_node.Name('a'),
                                  create_node.Constant('message', "'"))
        string = 'assert a, "message"\n'
        self._verify_match(node,string)
    def testNoMatchWithMessage(self):
        node = create_node.Assert(create_node.Name('a'),
                                  create_node.Constant('message', "'"))
        string = "assert a, 'message'\n"
        self._verify_match(node,string)
