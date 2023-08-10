from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ImportMatcherTest(BaseTestUtils):

    def testBasicMatchImport(self):
        string = 'import a\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testBasicMatchImport2(self):
        string = 'import a, b\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testBasicMatchImport3(self):
        string = 'import \t a,      b\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicMatchImport4(self):
        string = 'import a,b\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
