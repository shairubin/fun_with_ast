from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class NonlocalMatchTest(BaseTestUtils):

    def testSimple(self):
        string = "nonlocal x,y,z"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
