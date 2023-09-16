import unittest

import pytest
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class JoinStrMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.JoinedStr([create_node.Constant('X', "'")])
        string = "f'X'"
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())


    def testBasicMatchFromInput(self):
        node = GetNodeFromInput("f'X'")
        string = "(f'X')"
        #self._assert_match(node.value, string)
        self._verify_match(node, string)


    def testBasicFormatedValue(self):
        node = create_node.JoinedStr([create_node.FormattedValue(create_node.Name('a'))])
        string = "f'{a}'"
        self._assert_match(node, string)

    def testBasicFormatedValueDoubleQ(self):
        node = create_node.JoinedStr([create_node.FormattedValue(create_node.Name('a'))])
        string = "f\"{a}\""
        self._assert_match(node, string)

    def testMatchStringsAndFormatedValue(self):
        node = create_node.JoinedStr([
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Constant('x', "'")])
        string = "f'{a}x'"
        self._assert_match(node, string)


    def testMatchStringsAndFormatedValue2(self):
        node = create_node.JoinedStr([create_node.Constant('y', "'"),
                                      create_node.FormattedValue(create_node.Name('a'))])
        string = "f'y{a}'"
        self._assert_match(node, string)
    def testBasicMatchDoubleQuote2(self):
        node = create_node.JoinedStr([create_node.Constant("X", "\"")])
        string = "f\"X\""
        self._assert_match(node, string)

    def testMatchStringsAndFormatedValue3(self):
        node = create_node.JoinedStr([create_node.Constant('y', "'"),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.FormattedValue(create_node.Name('b'))])
        string = "f'y{a}{b}'"
        self._assert_match(node, string)


    def testMatchStringsAndFormatedValue4(self):
        node = create_node.JoinedStr([create_node.Constant('y',"'"),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Constant('x', "'"),
                                      create_node.FormattedValue(create_node.Name('b')),
                                      create_node.Constant('z', "'")])
        string = "f'y{a}x{b}z'"
        self._assert_match(node, string)

    def testMatchStringsAndFormatedValue8(self):
        node = create_node.JoinedStr([create_node.Constant('y', "\""),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Constant('x', "\""),
                                      create_node.FormattedValue(create_node.Name('b')),
                                      create_node.Constant('z', "\"")])
        string = "f\"y{a}x{b}z\""
        self._assert_match(node, string)


    def testMatchStringsAndFormatedValue5(self):
        node = create_node.JoinedStr([create_node.Constant('y', "\""),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Constant('\'', "\""),
                                      create_node.FormattedValue(create_node.Name('b')),
                                      create_node.Constant('z', "\"")])
        string = "f\"y{a}'{b}z\""
        self._assert_match(node, string)


    def testNoMatchStringsAndFormatedValue8(self):
        node = create_node.JoinedStr([create_node.Constant('y', "\""),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Constant('x', "'"),
                                      create_node.FormattedValue(create_node.Name('b')),
                                      create_node.Constant('z', "\"")])
        string = "f\"y{a}x{b}z\""
        self._assert_match(node, string)


    def _assert_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)
    def testNoMatchStringsAndFormatedValue5(self):
        node = create_node.JoinedStr([create_node.Constant('y', "'"),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Constant('\'', "\""),
                                      create_node.FormattedValue(create_node.Name('b')),
                                      create_node.Constant('z', "\"")])
        string = "f\"y{a}'{b}z\""
        self._assert_match(node, string)


# NEGATIVE TESTS - NO MATCH
    def testNoMatchStringsAndFormatedValueDQuote(self):
        node = create_node.JoinedStr([
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Constant('x', "\"")])
        string = "f'{a}x'"
        self._assert_match(node, string)


    def testBasicNoMatch(self):
        node = create_node.JoinedStr([create_node.Constant('X', "'")])
        string = "f'X '"
        self._assert_no_match(node, string)

    def testBasicNoMatch2(self):
        node = create_node.JoinedStr([create_node.Constant('X', "\"")])
        string = "f\"X \""
        self._assert_no_match(node, string)


    def testBasicnNMatchDoubleQuote3(self):
        node = create_node.JoinedStr([create_node.Constant('fun_with-ast', "\"")])
        string = "f\"fun-with-ast\""
        self._assert_no_match(node, string)

    def testBasicnNoMatchEmpty(self):
        node = create_node.JoinedStr([create_node.Constant('', "\"")])
        string = "f\"\""
        matcher = GetDynamicMatcher(node)
        with pytest.raises(ValueError):
            matcher.do_match(string)


    def testNoMatchStringsAndFormatedValue(self):
        node = create_node.JoinedStr([create_node.Constant('y', "'"),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.FormattedValue(create_node.Name('b'))])
        string = "f'y{a}{c}'"
        matcher = GetDynamicMatcher(node)
        self._assert_no_match(node, string)

    def testNoMatchFormatValue(self):
        node = create_node.JoinedStr([create_node.FormattedValue(create_node.Name('a'))])
        string = "f'  {a}'  "
        self._assert_no_match(node, string)

    def _assert_no_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)

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
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)

    def testBasicMatchFromInput53(self):
        node = GetNodeFromInput("f\"X\"")
        string = "(f\"X\")"
        self._verify_match(node, string)

    def testBasicMatchFromInput54(self):
        node = GetNodeFromInput("f'{X}'")
        string = "f'{X}'"
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)

    def testBasicMatchFromInput55(self):
        node = GetNodeFromInput("f\"{X}\"")
        string = "f\"{X}\""
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)

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
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)

    def testBasicMatchFromInput41(self):
        node = GetNodeFromInput("f\"Unknown norm type {type}\"")
        string = "(f\"Unknown norm type {type}\")"
        self._verify_match(node, string)


    def testBasicMatchFromInput2(self):
        node = GetNodeFromInput("f'X{a}'")
        string = "f'X{a}'"
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)
    def testBasicMatchFromInput3(self):
        node = GetNodeFromInput("f'X{a}[b]'")
        string = "f'X{a}[b]'"
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)

    def testBasicMatchFromInputNewLine(self):
        node = GetNodeFromInput("f'X{a}[b]'")
        string = "f'X{a}[b]\n'"
        self._assert_no_match(node.value, string)

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
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)
    def testMatchMultilLine14(self):
        node = GetNodeFromInput("f'XY'")
        string = "f'X'\nf'Y'"
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)

    def testMatchMultilLine13(self):
        node = GetNodeFromInput("f'XY'")
        string = "f'XY'"
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)

    def testMatchMultilLine2(self):
        node = GetNodeFromInput("f'X'")
        string = "f'X'    "
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)

    def testMatchPlaceholderEndOfString(self):
        string = """f\"FlaxBartEncoderLayer_{i}\""""
        node = GetNodeFromInput(string)
        self._verify_match(node.value, string)
        self._assert_match(node.value, string)

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
