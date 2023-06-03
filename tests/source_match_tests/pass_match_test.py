from fun_with_ast.manipulate_node import create_node
from tests.source_match_tests.base_test_utils import BaseTestUtils


class PassMatcherTest(BaseTestUtils):
    def testSimplePass(self):
        node = create_node.Pass()
        string = 'pass'
        self._verify_match(node, string)

    def testPassWithWS(self):
        node = create_node.Pass()
        string = '   \t pass  \t  '
        self._verify_match(node, string)

    def testPassWithWSAndComment(self):
        node = create_node.Pass()
        string = '   \t pass  \t #comment \t '
        self._verify_match(node, string)
