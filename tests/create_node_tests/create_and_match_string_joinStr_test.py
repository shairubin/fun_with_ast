import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class JoinStrMatcherTestCreateNode(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.JoinedStr([create_node.Constant('X', "'")])
        string = "f'X'"
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())
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
        with pytest.raises(Exception):
            matcher.do_match(string)
