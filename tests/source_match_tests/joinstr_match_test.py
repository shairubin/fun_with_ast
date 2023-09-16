import unittest

import pytest
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils




class JoinStrMatcherTests(BaseTestUtils):
    def testBasicMatchFromInput(self):
        node = GetNodeFromInput("f'X'")
        string = "(f'X')"
        #self._assert_match(node.value, string)
        self._verify_match(node, string)
###############################################################
# From Input tests
###############################################################
    def testBasicMatchFromInput5(self):
        node = GetNodeFromInput("f'X'")
        string = "f'X'"
        self._verify_match(node.value, string)

    def testBasicMatchFromInput51(self):
        node = GetNodeFromInput("f'X'")
        string = "(f'X')"
        self._verify_match(node, string)

    def testBasicMatchFromInput52(self):
        node = GetNodeFromInput("f\"X\"")
        string = "f\"X\""
        self._verify_match(node, string)

    def testBasicMatchFromInput53(self):
        node = GetNodeFromInput("f\"X\"")
        string = "(f\"X\")"
        self._verify_match(node, string)

    def testBasicMatchFromInput54(self):
        node = GetNodeFromInput("f'{X}'")
        string = "f'{X}'"
        self._verify_match(node, string)

    def testBasicMatchFromInput55(self):
        node = GetNodeFromInput("f\"{X}\"")
        string = "f\"{X}\""
        self._verify_match(node, string)

    def testBasicMatchFromInput56(self):
        node = GetNodeFromInput("f\"{X}\"")
        string = "(f\"{X}\")"
        self._verify_match(node, string)

    def testBasicMatchFromInput57(self):
        node = GetNodeFromInput("f'{X}'")
        string = "(f'{X}')"
        self._verify_match(node, string)

    def testBasicMatchFromInput4(self):
        node = GetNodeFromInput("f\"Unknown norm type {type}\"")
        string = "f\"Unknown norm type {type}\""
        self._verify_match(node, string)

    def testBasicMatchFromInput41(self):
        node = GetNodeFromInput("f\"Unknown norm type {type}\"")
        string = "(f\"Unknown norm type {type}\")"
        self._verify_match(node, string)


    def testBasicMatchFromInput2(self):
        node = GetNodeFromInput("f'X{a}'")
        string = "f'X{a}'"
        self._verify_match(node, string)
    def testBasicMatchFromInput3(self):
        node = GetNodeFromInput("f'X{a}[b]'")
        string = "f'X{a}[b]'"
        self._verify_match(node, string)

    def testBasicMatchFromInputNewLine(self):
        node = GetNodeFromInput("f'X{a}[b]'")
        string = "f'X{a}[b]\n'"
        self._verify_match(node, string)

    def testMatchMultilLine(self):
        with pytest.raises((SyntaxError)):
            node = GetNodeFromInput("f'X\n'")
    def testMatchMultilLine1(self):
        node = GetNodeFromInput("f'X'")
        string = "(f'X')"
        self._verify_match(node, string)
    def testMatchMultilLine11(self):
        node = GetNodeFromInput("f'XY'")
        string = "(f'X'\nf'Y')"
        self._verify_match(node, string)

    def testMatchMultilLine12(self):
        node = GetNodeFromInput("f'XY'")
        string = "f'X'\nf'Y'"
        self._verify_match(node, string)
    def testMatchMultilLine14(self):
        node = GetNodeFromInput("f'XY'")
        string = "f'X'\nf'Y'"
        self._verify_match(node, string)

    def testMatchMultilLine13(self):
        node = GetNodeFromInput("f'XY'")
        string = "f'XY'"
        self._verify_match(node, string)

    def testMatchMultilLine2(self):
        node = GetNodeFromInput("f'X'")
        string = "f'X'    "
        self._verify_match(node, string)

    def testMatchPlaceholderEndOfString(self):
        string = """f\"FlaxBartEncoderLayer_{i}\""""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    @pytest.mark.skip('not supported yet')
    def testMatchComment(self):
        node = GetNodeFromInput("f'X'")
        string = "f'X'   # comment "
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)

    def testJstrWitJoinedStr62(self):
        string = """(f\"A_{i}\",)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrWitJoinedStr63(self):
        string = """f\"A_{i}\""""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
