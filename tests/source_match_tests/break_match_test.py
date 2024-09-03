import sys

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class BreakTest(BaseTestUtils):

    def testSimple(self):
        string = """if __name__ == "__main__":
    while retry > 0:
        if build_result != "succeeded":
            retry = retry - 1
        else:
            break"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testSimple2(self):
        string = """if __name__ == "__main__":
    while retry > 0:
        if build_result != "succeeded":
            retry = retry - 1
        else:
            break
            """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    @pytest.mark.skipif(sys.version_info.major == 3 and sys.version_info.minor == 10, reason="requires python3.10 or higher")
    def testexamplefail_311_only(self):
        pass

