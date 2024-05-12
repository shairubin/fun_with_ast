import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class WithTest(BaseTestUtils):

    def testWithBasic(self):
        string = 'with a as fp:\n   pass'
        node =GetNodeFromInput(string)
        self._verify_match(node, string)

    def testWithBasic2(self):
        string = 'with a as fp:\n   pass\n   pass\n   pass'
        node =GetNodeFromInput(string)
        self._verify_match(node, string)

    def testWithBasic3(self):
        string = 'with a as fp:\n   pass\n   pass\n   pass\n  '
        node =GetNodeFromInput(string)
        self._verify_match(node, string, trim_suffix_spaces=True)

    def testWithBasic4(self):
        string = 'with a as fp:\n   pass\n   pass\n   pass\n  '
        node =GetNodeFromInput(string)
        with pytest.raises(AssertionError):
            self._verify_match(node, string, trim_suffix_spaces=False)

    def testWithCompound(self):
        string = 'with a as ap, b as bp:\n   pass\n   pass\n   pass\n  '
        node =GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)
