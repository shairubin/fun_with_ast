from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class DelMatcherTest(BaseTestUtils):

    def testSimpleDel(self):
        string = "del a"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testSimpleDel2(self):
        string = "del \t a"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testSimpleDel3(self):
        string = "del \t a \t \n  "
        node = GetNodeFromInput(string)
        self._verify_match(node, string, trim_suffix_spaces=True)
    def testSimpleDel3_1(self):
        string = "del \t a \t "
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)
    def testSimpleDel4(self):
        string = "del \t a \t #comment "
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testListDel(self):
        string = "del a, b"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testListDel2(self):
        string = "del a, b # comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
