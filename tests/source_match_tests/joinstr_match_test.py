import unittest

import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.dynamic_matcher import GetDynamicMatcher


class JoinStrMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.JoinedStr([create_node.Str('X')])
        string = "f'X'"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicNoMatch(self):
        node = create_node.JoinedStr([create_node.Str('X')])
        string = "f'X '"
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)

    def testBasicMatchDoubleQuote(self):
        node = create_node.JoinedStr([create_node.Str("X")])
        string = "f\"X\""
        matcher = GetDynamicMatcher(node)
        self._assert_match(matcher, string)

    def testBasicnNMatchDoubleQuote(self):
        node = create_node.JoinedStr([create_node.Str('fun_with-ast')])
        string = "f\"fun-with-ast\""
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)

    def testBasicnNoMatchEmpty(self):
        node = create_node.JoinedStr([create_node.Str('')])
        string = "f\"\""
        matcher = GetDynamicMatcher(node)
        with pytest.raises(NotImplementedError):
            matcher.Match(string)

    def testBasicFormatedValue(self):
        node = create_node.JoinedStr([create_node.FormattedValue(create_node.Name('a'))])
        string = "f'{a}'"
        matcher = GetDynamicMatcher(node)
        self._assert_match(matcher, string)

    def testBasicFormatedValueDoubleQ(self):
        node = create_node.JoinedStr([create_node.FormattedValue(create_node.Name('a'))])
        string = "f\"{a}\""
        matcher = GetDynamicMatcher(node)
        self._assert_match(matcher, string)

    def testNoMatchFormatValue(self):
        node = create_node.JoinedStr([create_node.FormattedValue(create_node.Name('a'))])
        string = "f'  {a}'  "
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)

    def testMatchStringsAndFormatedValue(self):
        node = create_node.JoinedStr([
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Str('x')])
        string = "f'{a}x'"
        matcher = GetDynamicMatcher(node)
        self._assert_match(matcher, string)

    def testMatchStringsAndFormatedValue2(self):
        node = create_node.JoinedStr([create_node.Str('y'),
                                      create_node.FormattedValue(create_node.Name('a'))])
        string = "f'y{a}'"
        matcher = GetDynamicMatcher(node)
        self._assert_match(matcher, string)

    def testMatchStringsAndFormatedValue3(self):
        node = create_node.JoinedStr([create_node.Str('y'),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.FormattedValue(create_node.Name('b'))])
        string = "f'y{a}{b}'"
        matcher = GetDynamicMatcher(node)
        self._assert_match(matcher, string)

    def testNoMatchStringsAndFormatedValue(self):
        node = create_node.JoinedStr([create_node.Str('y'),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.FormattedValue(create_node.Name('b'))])
        string = "f'y{a}{c}'"
        matcher = GetDynamicMatcher(node)
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)

    def testMatchStringsAndFormatedValue4(self):
        node = create_node.JoinedStr([create_node.Str('y'),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Str('x'),
                                      create_node.FormattedValue(create_node.Name('b')),
                                      create_node.Str('z')])
        string = "f'y{a}x{b}z'"
        matcher = GetDynamicMatcher(node)
        self._assert_match(matcher, string)

    def testMatchStringsAndFormatedValue5(self):
        node = create_node.JoinedStr([create_node.Str('y'),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Str('x'),
                                      create_node.FormattedValue(create_node.Name('b')),
                                      create_node.Str('z')])
        string = "f\"y{a}x{b}z\""
        matcher = GetDynamicMatcher(node)
        self._assert_match(matcher, string)


    def testMatchStringsAndFormatedValue5(self):
        node = create_node.JoinedStr([create_node.Str('y'),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Str('\''),
                                      create_node.FormattedValue(create_node.Name('b')),
                                      create_node.Str('z')])
        string = "f\"y{a}'{b}z\""
        matcher = GetDynamicMatcher(node)
        self._assert_match(matcher, string)


    def testNoMatchStringsAndFormatedValue5(self):
        node = create_node.JoinedStr([create_node.Str('y'),
                                      create_node.FormattedValue(create_node.Name('a')),
                                      create_node.Str('\''),
                                      create_node.FormattedValue(create_node.Name('b')),
                                      create_node.Str('z')])
        string = "f\"y{a}'{b}z \""
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)


    def _assert_match(self, matcher, string):
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)
