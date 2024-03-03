import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.base_matcher import SourceMatcher
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


@pytest.fixture(autouse=True)
def run_around_tests(): # TODO not very smart global variable
    SourceMatcher.parentheses_stack.reset()
    yield

class ConstantNumMatcherTest(BaseTestUtils):


    def testBasicMatchNum(self):
        node = create_node.Num('1')
        string = '1'
        self._assert_match(node, string)

    def testBasicMatchNone(self):
        node = create_node.CreateNone('None')
        string = 'None'
        self._assert_match(node, string)

    def _assert_match(self, node, string):
        self._verify_match(node, string)
    def testBasicMatchNumBinary(self):
        node = create_node.Num('0b0')
        string = '0b0'
        self._assert_match(node, string)

    def testBasicMatchNumSci(self):
        node = create_node.Num('1e-06')
        string = '1e-06'
        self._assert_match(node, string)


    def testBasicMatchNumSci2(self):
        node = create_node.Num('0.2')
        string = '0.2'
        self._assert_match(node, string)

    def testBasicMatchNumSci3(self):
        node = create_node.Num('123456789.98765433')
        string = '123456789.98765433'
        self._assert_match(node, string)

    def testBasicMatchNumWithError(self):
        node = create_node.Num('1')
        string = '1:#Comment'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)


    @pytest.mark.skip('Not implemented yet')
    def testBasicMatchWithPlusSign3(self):
        node = create_node.Num('1')
        string = '(    (    +1   )   ) # comment'
        self._assert_match(node, string)

    def testBasicMatchWithMinusSign(self):
        node = create_node.Num('-1')
        string = '  -1   \t\n'
        #with pytest.raises(BadlySpecifiedTemplateError):
        self._assert_match(node, string)

    def testBasicMatchWithdWS(self):
        node = create_node.Num('1')
        string = '1   '
        self._verify_match(node, string)


    def testBasicMatchWithdWSExpr(self):
        node = create_node.Expr( create_node.Num('1'))
        string = '   1   \n'
        self._assert_match(node, string)

    def testMatchWSWithComment(self):
        node = create_node.Num('1')
        string = '   1   #comment'
        self._assert_match(node, string)

    def testWithParans(self):
        node = create_node.Num('1')
        string = '(1)'
        self._assert_match(node, string)

    def testWithParansFloat(self):
        node = create_node.Num('1.123')
        string = '(1.123)'
        self._assert_match(node, string)

    def testWithParansAndWS(self):
        node = create_node.Num('1')
        string = '(   1   )     '
        self._assert_match(node, string)

    def testWithMultiParansAndWS(self):
        node = create_node.Num('-1')
        string = '((   -1   )    ) '
        self._verify_match(node, string)


    def testNoMatchMultiParansAndWS(self):
        node = create_node.Num('1')
        string = '((   1   )     '
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)
    def testNoMatchMultiParansAndWS2(self):
        node = create_node.Num('1')
        string = '(   1   )     )'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)

    def testLargeNumberMatch(self):
        node = create_node.Num('1234567890987654321')
        string = '1234567890987654321'
        self._verify_match(node, string)

    def testLargeNumberMatchHex(self):
        node = create_node.Num('0xffaa')
        string = '\t0xffaa\t #comment'
        self._verify_match(node, string)

    def testLargeNumberMatchHexNoMatch(self):
        node = create_node.Num('0xffab')
        string = '\t0xffaa\t #comment'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)

    def testBasicNoMatchNum(self):
        node = create_node.Num('2')
        string = '1'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)
        string = '2'
        self._verify_match(node, string)

    def testMatchBool(self):
        node = create_node.Bool(False)
        string = 'False'
        self._verify_match(node, string)

    def testMatchBoolParans(self):
        node = create_node.Bool(False)
        string = '(False)'
        self._verify_match(node, string)


    @pytest.mark.skip('Not implemented yet -- issue #73')
    def testStringWithDQ(self):
        node = create_node.Str("\"0xffaa\"")
        string = "\"0xffaa\""
        self._verify_match(node, string)
    @pytest.mark.skip('Not implemented yet -- issue #73')
    def testStringWithSQ(self):
        node = create_node.Str("'0xffaa'")
        string = "'0xffaa'"
        self._verify_match(node, string)
#############################################################
#Test From Input
###########################################################
    def testStringFronInput(self):
        string = "'0xffaa'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testStringFronInput2(self):
        string = "'   0xffaa'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringFronInput2_1(self):
        string = '1'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testStringFronInput2_11(self):
        string = '(1.012345)'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringFronInput2_31(self):
        string = '1 #comment '
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringFronInput2_2(self):
        string = '1\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringFronInput2_3(self):
        string = '1 # comment \n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringFronInput2_4(self):
        string = '1 # comment \n   '
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testStringFronInput2_41(self):
        string = '1 # comment \n   '
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string)


    def testStringFronInput2_5(self):
        string = '1\n   '
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringFronInput2_6(self):
        string = '1\n   '
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    def testStringFronInput3(self):
        string = "if True:\n   'fun-with-ast'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testEllipsis(self):
        string = "..."
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNumSci1(self):
        string = "1e-6"
        node = GetNodeFromInput(string)
        self._verify_match(node, string, get_source_after_reset=False)

    def testNumSci2(self):
        string = "-9.99E+9"
        node = GetNodeFromInput(string)
        self._verify_match(node, string, get_source_after_reset=False)

    def testNumSci3(self):
        string = "4.2e-06"
        node = GetNodeFromInput(string)
        self._verify_match(node, string, get_source_after_reset=False)

    def testNumSci4(self):
        string = "-.2E-4"
        node = GetNodeFromInput(string)
        self._verify_match(node, string, get_source_after_reset=False)

    def testNumSci5(self):
        string = "4e6660"
        node = GetNodeFromInput(string)
        self._verify_match(node, string, get_source_after_reset=False)

    def testNumSci6(self):
        string = "4e6"
        node = GetNodeFromInput(string)
        self._verify_match(node, string, get_source_after_reset=False)

    def testNumSci6_1(self):
        string = "(4e6)"
        node = GetNodeFromInput(string)
        self._verify_match(node, string, get_source_after_reset=False)


    @pytest.mark.skip('issue 266')
    def testNumSciEndOfCall(self):
        string = 'npt.assert_allclose(workspace.blobs[output], ref(), rtol=1e-3)'
        node = GetNodeFromInput(string)
        self._verify_match(node, string, get_source_after_reset=False)

    @pytest.mark.skip('issue 266')
    def testNumSciEndOfCall1(self):
        string = 'allclose(rtol=1e-3)'
        node = GetNodeFromInput(string)
        self._verify_match(node, string, get_source_after_reset=False)
    @pytest.mark.skip('issue 266')
    def testNumSciEndOfCall2(self):
        string = '(1e-3)'
        node = GetNodeFromInput(string)
        self._verify_match(node, string, get_source_after_reset=False)

    def testComplex(self):
        string = "a=1+2j"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testComplex2(self):
        string = "a=1 + 2j"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testComplex3(self):
        string = "a=1 + 2j # comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testComplex4(self):
        string = "a=1 -2j # comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testComplex5(self):
        string = "a=1.23 -2j # comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testComplex6(self):
        string = "a=-2j # comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
