import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput, FailedToCreateNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class NonlocalMatchTest(BaseTestUtils):

    def testSimple(self):
        string = "global x,y,z"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testComment(self):
        string = "global x,y,z # comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSingle(self):
        string = "global x"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSingleWithComma(self):
        string = "global x,"
        with pytest.raises(FailedToCreateNodeFromInput):
            GetNodeFromInput(string)

    def testList(self):
        string = "global [x,y,z]"
        with pytest.raises(FailedToCreateNodeFromInput):
            GetNodeFromInput(string)


# non-local
    def testNonLocalSimple(self):
        string = "nonlocal x,y,z"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNonLocalComment(self):
        string = "nonlocal \t x,y,z # comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNonLocalSingle(self):
        string = "nonlocal     x"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNonLocalWithComma(self):
        string = "nonlocal x,"
        with pytest.raises(FailedToCreateNodeFromInput):
            GetNodeFromInput(string)

    def testNonLocalList(self):
        string = "nonlocal [x,y,z]"
        with pytest.raises(FailedToCreateNodeFromInput):
            GetNodeFromInput(string)



