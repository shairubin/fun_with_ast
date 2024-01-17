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
    def testBasicMatchFromInput56_1(self):
        node = GetNodeFromInput("f\"{X }\"")
        string = "(f\"{X }\")"
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
        node = GetNodeFromInput("f'XY'")
        string = "(f'XY')"
        self._verify_match(node, string)

    def testMatchMultilLine11(self):
        node = GetNodeFromInput("(f'X'\nf'Y')")
        string = "(f'X'\nf'Y')"
        self._verify_match(node, string)

    def testMatchMultilLine12(self):
        node = GetNodeFromInput("(f'X'\nf'Y')", get_module=True)
        string = "(f'X'\nf'Y')"
        self._verify_match(node, string)
    def testMatchMultilLine14(self):
        node = GetNodeFromInput("f'X'\nf'Y'")
        string = "f'X'\nf'Y'"
        with pytest.raises(ValueError, match=r'.*jstr string not in call_args context.*'):
            self._verify_match(node, string)

    def testMatchMultilLine13(self):
        node = GetNodeFromInput("f'XY'")
        string = "f'XY'"
        self._verify_match(node, string)

    @pytest.mark.skip("not supported yet")
    def testMatchMultilLine15(self):
        node = GetNodeFromInput("f'XY'")
        string = "f'X'f'Y'"
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
        node = GetNodeFromInput(string, get_module=True) # note that these are 3! different strings
        with pytest.raises(ValueError, match=r'.*jstr string not in call_args context.*') :
            self._verify_match(node, string)

    def testJstrWithsLinesAndParamsAndParen(self):
        string = """(f"{opname}: operator. "
f"The '{module}' "
f"Python  {context}") """
        node = GetNodeFromInput(string, get_module=True)  # note that thisd is 1 (one)! jstr string
        self._verify_match(node, string)

    def testJstrWithsLinesAndParamsAndParen2(self):
        string = """(f"{opname}: operator. "
f"'{module}'") """
        node = GetNodeFromInput(string, get_module=True)  # note that this is 1 (one)! jstr string
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

    def testJstrWithsLinesNoF_Prefix(self):
        string = """msg = (
            f"Can't get source for {obj}. TorchScript requires source access in "
            "order to carry out compilation, make sure original .py files are "
            "available."
        )
  """
    def testJstrWithsLinesNoF_Prefix0_1(self):
        string = """msg = (
            f"Can't get source for {obj}. TorchScript requires source access in "
            "order to carry out compilation, make sure original .py files are "
            "available."
        )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrWithsLinesNoF_Prefix1(self):
        string = """msg = (f"C"\n"o"\n"a")
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrWithsLinesNoF_Prefix2(self):
        string = """msg = (
        f"C"
        "o"
        "a")"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrWithsLinesNoF_Prefix3(self):
        string = """msg = (
        f"C"
        "o"
        )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrWithsLinesNoF_Prefix3_1(self):
        string = """msg = (\nf"C"\n)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrWithsLinesNoF_Prefix3_2(self):
        string = """msg = (f"C")"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrWithsLinesNoF_Prefix3_3(self):
        string = """msg = f\"C\""""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrWithsLinesNoF_Prefix3_4(self):
        string = """msg = (
        f"C"
        f"o"
        )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrFindQuoteInaSingleString(self):
        string = """
print(f"Exporting labels for {args.org}/{args.repo}")
obj = boto3.resource("s3").Object("ossci-metrics", labels_file_name)
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrFindQuoteInaSingleString2(self):
        string = """f"{args.org}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrMixedFTypes(self):
        string = """('could not identify license file '
                                     f'for {root}')"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrMixedFTypes1(self):
        string = """('could not identify license file '
                                     'for {root}') """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string) # note that this is a module with ONE strings
    def testJstrMixedFTypes3(self):
        string = """(f'could not identify license file '
f'for {root}') """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)



    def testJstrMixedFTypes3_02(self):
        string = """(f'X '
'Y{W}') """ # please note that this part is a regular string and NOT a jstr string, henc this is not supported at this time
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrMixedFTypes3_03(self):
        string = """(f'X '
f'Y{W}') """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrMixedFTypes3_04(self):
            string = """(f'X '\n     f'Y{W}') """
            node = GetNodeFromInput(string, get_module=True)
            self._verify_match(node, string)

    def testJstrMixedFTypes4(self):
        string = """\"could not identify license file \"
f\"for {root}\""""
        node = GetNodeFromInput(string, get_module=True) # this is a module with TWO strings
        self._verify_match(node, string)

    def testJstrMixedFTypes4_01(self):
        string = """(\"could not identify license file \"
f\"for {root}\")"""
        node = GetNodeFromInput(string, get_module=True)  # this is a module with One strings
        self._verify_match(node, string)

    def testJstrMixedFTypes4_02(self):
        string = """(\"could not identify license file \"
f\"for {root}\")"""
        node = GetNodeFromInput(string, get_module=False)  # this is an expression with One strings
        self._verify_match(node, string)

    def testJstrMixedFTypes4_1(self):
        string = """\"X \"\nf\"Y{root}\" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrMixedFTypes4_2(self):
        string = """\"X \"\nf\"Y\" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrMixedFTypes4_2_1(self):
        string = """(\"X \"\nf\"Y\" )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrMixedFTypes4_2_2(self):
        string = """(f\"X \"\nf\"Y\" )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrMixedFTypes4_2_3(self):
        string = """(\"X \"\n\"Y\" )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testJstrMixedFTypes4_3(self):
        string = """(\"X \"
                                     f\"Y\"
                                     \"Z\" ) """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrMixedFTypes4_3_1(self):
        string = """(\"X \"\nf\"Y\"\n\"Z\" ) """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrMixedFTypes4_3_2(self):
        string = """(\"X \"\nf\"Y\"    \n\"Z\" ) """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrMixedFTypes4_3_3(self):
        string = """(\"X \"    \n\"Z\" ) """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrMixedFTypes4_3_4(self):
        string = """(f\"Y\"       \n\"Z\" ) """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrMixedFTypes4_3_5(self):
        string = """(f\"Y\"\n\"Z\" ) """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrWithConversion(self):
        string = """f"module {__name__!r} has no attribute {name!r}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrWithConversion2(self):
        string = """f"module {__name__!r}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrWithConversion3(self):
        string = """f"{abc!r}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testJstrWithConversion4(self):
        string = """f"{abc!s}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testJstrWithConversion5(self):
        string = """f"{abc!a}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testJstrWithConversion5_1(self):
        string = """f"{abc !a}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testJstrWithConversion5_2(self):
        string = """f"{   abc !a}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testJstrWithConversion5_3(self):
        string = """f"{7!a}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testJstrWithConversion5_4(self):
        string = """f"{ 7   !a}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testJstrWithConversion5_5(self):
        string = """f"{ 7!a}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    def testJstrWithConversion6(self):
        string = """f"{abc!c}" """
        with pytest.raises(SyntaxError):
            node = GetNodeFromInput(string, get_module=True)
