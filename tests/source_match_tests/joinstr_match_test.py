import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput, FailedToCreateNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
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
        with pytest.raises((FailedToCreateNodeFromInput)):
            node = GetNodeFromInput("f'\''") # not supported in python

    def testBasicMatchFromInput(self):
        node = GetNodeFromInput("f'X'")
        string = "(f'X')"
        self._verify_match(node, string)
    def testBasicMatchFromInput5(self):
        node = GetNodeFromInput("f'X'")
        string = "f'X'"
        self._verify_match(node, string)

    def testBasicNoMatchFromInput5(self):
        node = GetNodeFromInput("f'X'")
        string = "f'Y'"
        with pytest.raises(BadlySpecifiedTemplateError):
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
        string = "f\"Unknown norm type {tpe}\""
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)
    def testBasicMatchFromInput41(self):
        node = GetNodeFromInput("f\"Unknown norm type {type}\"")
        string = "(f\"Unknown norm type {type}\")"
        self._verify_match(node, string)

    def testBasicMatchFromInput2(self):
        node = GetNodeFromInput("f'X{a}'")
        string = "f'X{a}'"
        self._verify_match(node, string)
        string = "f\"X{b}\""
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testBasicMatchFromInput3(self):
        node = GetNodeFromInput("f'X{a}[b]'")
        string = "f'X{a}[b]'"
        self._verify_match(node, string)
        string = "f\"X{a}[b]\""
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)


    def testBasicMatchFromInputNewLine(self):
        with pytest.raises(FailedToCreateNodeFromInput):
            node = GetNodeFromInput("f'X{a}\n[b]'")

    def testMatchMultilLine(self):
        with pytest.raises((FailedToCreateNodeFromInput)):
            node = GetNodeFromInput("f'X\n'")
    def testMatchMultilLine1NoMatch(self):
        node = GetNodeFromInput("f'XY'")
        string = "(f'X Y')"
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)
    def testMatchMultilLine1Match(self):
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
        string = "f\"YX\""
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    @pytest.mark.skip("not supported yet")
    def testMatchMultilLine15(self):
        node = GetNodeFromInput("f'XY'")
        string = "f'X'f'Y'"
        self._verify_match(node, string)

    def testMatchMultilLine2(self):
        node = GetNodeFromInput("f'X'", get_module=True)
        string = "f'X'    "
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
        string = """(f"{opname}: operator. "
f"Th '{module}' "
f"Python  {context}") """
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)
    def testJstrWithsLinesAndParamsAndParen2(self):
        string = """(f"{opname}: operator. "
f"'{module}'") """
        node = GetNodeFromInput(string, get_module=True)  # note that this is 1 (one)! jstr string
        self._verify_match(node, string)

    def testJstrMultipleParts(self):
        string = """(f"{opname}{abc}{xyz}") """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
        node2 = GetNodeFromInput(string)
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node2, string.replace('x', 'y'))

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
        string = """msg = (
        f"C"
        "i"
        "a")"""
        with pytest.raises(BadlySpecifiedTemplateError):
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

    def testJstrMixedFTypes01(self):
        string = """('first line '
                                     f'second line ')"""
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
        string = """f"module {__name__!a}" """
        with pytest.raises(BadlySpecifiedTemplateError):
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
        string = """f"{ 7  !a}" """
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testJstrWithConversion6(self):
        string = """f"{abc!c}" """
        with pytest.raises(FailedToCreateNodeFromInput):
            node = GetNodeFromInput(string, get_module=True)

    def testModule7Partial(self):
        string =  """new_k = f"{k[name_idx][:-1]}_{i}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testModule7Partial2(self):
        string =  """f"{k[name_idx][:-1]}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testModule7Partial3(self):
        string =  """f"{k[:-1]}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testModule18Partial(self):
        string =  """f"-DPYTHON_INCLUDE_DIR={sysconfig.get_path('include')}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testModule18Partial2(self):
        string =  """f"{a.b('include')}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testDoubleLineCausesResetMissing(self):
        string =  """(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads}).") """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testSubscriptWithConstant(self):
        string =  """f"{c['Name']}\\n" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testSubscriptWithConstant2(self):
        string =  "f\"{c['Name']}\\n\""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSubscriptWithConstant3(self):
        string =  "f\"{c['Name']}\\n\" "
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBinOp(self):
        string = "f\"number of nodes not the same {old_num_nodes - delta}, {new_num_nodes}\\n {fx_g.graph} \\n {new_graph}\""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testListComprehension(self):
        string = "f\"{[x for x in range(10)]}\""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testQuotesInJstStr(self):
        string = """f'__load_module("{self.index.module}").{self.index.qualname}'"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testQuotesInJstStr2(self):
        string = """f'__a("{b}").{c}'"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testQuotesInJstStr3(self):
        string = "f\"test\" if run_url is not None else \"job\""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testQuotesInJstStr4(self):
        string = """"The"

        """
        node = GetNodeFromInput(string, get_module=False) # this is inconsistent with the
                                                          # fact that Expr should not match multiline
        self._verify_match(node, string, trim_suffix_spaces=True)
    def testQuotesInJstStr4_0(self):
        string = """"The"

        """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)

    def testQuotesInJstStr4_0_1(self):
        string = """"The"

        """
        node = GetNodeFromInput(string, get_module=False)
        with pytest.raises(AssertionError): # 'Expr node does not support trailing white spaces'
            self._verify_match(node, string, trim_suffix_spaces=False)

    @pytest.mark.skip("not supported yet - issue TBD")
    def testQuotesInJstStr4_1(self):
        string = """\"The\"\n\"problem\"
        """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testQuotesInJstStr4_5(self):
        string = """(
        f"The  If "
)
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testQuotesInJstStr5(self):
        string = """(
        f"The {args.action} {job_link} was canceled. If ")"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testQuotesInJstStr6(self):
        string = """
def main() -> None:
    msg = (
        f"The"     
    )
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testQuotesInJstStr7(self):
        string = """
(f"The"     
    )
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def test_bwd(self):
        string = """f"{suite_name}[{test_name}]:{'bwd' if bwd else 'fwd'}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def test_bwd2(self):
        string = """f"{suite_name}[{test_name}]:{'bwd' if bwd else 'fwd'}" """
        node = GetNodeFromInput(string, get_module=False)
        with pytest.raises(AssertionError): # 'Expr node does not support trailing white spaces'
            self._verify_match(node, string)

    def test_op_in_string(self):
        string = """f'_{index}' + '\\n'"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def test_op_in_string2(self):
        string = """f"_{index}" + '\\n'"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def test_op_in_string3(self):
        string = """f"_{index}" + "\\n" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def test_func_call(self):
        string = """f'_{index}'.add()"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def test_internal_quote(self):
        string = """f'unsupported autocast device_type \\'{dev}\\''"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def test_internal_quote1_1(self):
        string = """msg = f'unsupported autocast device_type \\'{dev}\\''"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def test_internal_quote2(self):
        string = """f"unsupported autocast device_type \\'{dev}\\'" """
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string, trim_suffix_spaces=True)

    def test_internal_quote2_1(self):
        string = """f"unsupported autocast device_type \\'{dev}\\'" """
        node = GetNodeFromInput(string, get_module=False)
        with pytest.raises(AssertionError): # 'Expr node does not support trailing white spaces'
            self._verify_match(node, string, trim_suffix_spaces=False)

    def test_internal_quote3(self):
        string = """f"unsupported autocast device_type \\"{dev}\\"" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)


    def test_internal_quote4(self):
        string = """f"my_mode '{mode}' is not supported" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)


    def test_JstrHTTP(self):
        string = "f'https://{cognito_domain}/oauth2/token'"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)

    @pytest.mark.skip("issue #328")
    def test_JstrEqualInVariable(self):
        string = "logger.info(f'{task_id=}')"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)

    def test_JstrTripleQuote(self):
        string = "f\"\"\"test\"\"\""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)
    def test_JstrTripleQuote2(self):
        string = "f\"test\""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)


    def test_JstrTripleQuote3(self):
        string = """f\"\"\"
                        SELECT endpoint
                        FROM {gpt_endpoint_table_name}
                        WHERE runtime = (SELECT MAX(runtime) FROM {gpt_endpoint_table_name} WHERE modelname='{model_name}') and modelname='{model_name}'
                        ORDER BY avgtime ASC
         \"\"\""""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)
    def test_JstrTripleQuote4(self):
        string = """f\"\"\"
                        SELECT endpoint
                        FROM test
         \"\"\""""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)

    def test_JstrTripleQuote4_1(self):
        string = """f\"\"\"
                      SELECT
                        \"\"\""""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)

    @pytest.mark.skip("issue #332")
    def test_JstrTripleQuote5(self):
        string = """f\"\"\"\
digraph G {{
rankdir = LR;
node [shape=box];
{edges}
}}
\"\"\"
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)
    @pytest.mark.skip("issue #332")
    def test_JstrTripleQuote5_1(self):
        string = """f\"\"\"\
digraph G {{
rankdir = LR;
}}
\"\"\"
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)
    def test_JstrTripleQuote5_2(self):
        string = """f' \
digraph G {{ \
rankdir = LR; \
}}'
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def test_JstrTripleQuote5_2_1(self):
        string = """f' \
digraph G  \
rankdir = LR; \
 \
'
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
