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
        string = "f'X'    " # WS at the end of line not supported
        with pytest.raises(AssertionError):
            self._verify_match(node, string)
    def testMatchMultilLine2_1(self):
        node = GetNodeFromInput("f'X'", get_module=True)
        string = "f'X'    \n"
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

    def testJstrWithsLinesAndParams(self):
        string = """f"{opname}: operator. "
f"The '{module}' "
f"Python  {context}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrWithsLinesAndParams2(self):
        string = """f"{opname}: operator. "\nf"The '{module}' "\nf"Python  {context}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrWithsLinesAndParams3(self):
        string = """f"X"\nf"Y"\nf"Z" """
        node = GetNodeFromInput(string, get_module=True) # node the module context these are not joined strings
        self._verify_match(node, string)
    def testJstrWithsLinesAndParams4(self):
        string = """a(f"X"\nf"Y"\nf"Z") """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrWithsLinesAndParams5(self):
        string = """a(f"X"
f"Y"
f"Z") """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrWithsLinesAndParams6(self):
        string = """a(f"X"
f"Y"
f"Z") # comment """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
