import unittest

import pytest
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from tests.source_match_tests.base_test_utils import BaseTestUtils




class JoinStrMatcherTests(BaseTestUtils):

    def testBasicMatchEmpty(self):
        node = GetNodeFromInput("f''")
        string = "(f'')"
        self._verify_match(node, string)
    def testBasicMatchEmpty1(self):
        node = GetNodeFromInput("f\"\"")
        string = "f\"\""
        self._verify_match(node, string)
    def testBasicMatchEmpty11(self):
        node = GetNodeFromInput("f\"\"")
        string = "f\'\'"
        self._verify_match(node, string)

    def testBasicMatchEmpty2(self):
        with pytest.raises((SyntaxError)):
            node = GetNodeFromInput("f'\''") # not supported in python

    def testBasicMatchFromInput(self):
        node = GetNodeFromInput("f'X'")
        string = "(f'X')"
        self._verify_match(node, string)
    def testBasicMatchFromInput5(self):
        node = GetNodeFromInput("f'X'")
        string = "f'X'"
        self._verify_match(node, string)

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
        with pytest.raises(SyntaxError):
            node = GetNodeFromInput("f'X{a}\n[b]'")

    def testMatchMultilLine(self):
        with pytest.raises((SyntaxError)):
            node = GetNodeFromInput("f'X\n'")
    def testMatchMultilLine1(self):
        node = GetNodeFromInput("f'X'")
        string = "(f'X')"
        self._verify_match(node, string)
    def testMatchMultilLine11(self):
        node = GetNodeFromInput("(f'X'\nf'Y')")
        string = "(f'X'\nf'Y')"
        self._verify_match(node, string)

    def testMatchMultilLine12(self):
        node = GetNodeFromInput("f'X'\nf'Y'", get_module=True)
        string = "f'X'\nf'Y'"
        self._verify_match(node, string)
    def testMatchMultilLine14(self):
        node = GetNodeFromInput("f'X'\nf'Y'")
        string = "f'X'\nf'Y'"
        with pytest.raises(AssertionError):
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

    def testMatchComment(self):
        node = GetNodeFromInput("f'X'")
        string = "f'X'   # comment "
        self._verify_match(node, string)

    def testJstrWitJoinedStr62(self):
        string = """(f\"A_{i}\",)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrWitJoinedStr63(self):
        string = """f\"A_{i}\""""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
