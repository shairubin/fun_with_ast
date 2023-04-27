import unittest

import pytest

import create_node
from fun_with_ast.exceptions_source_match import BadlySpecifiedTemplateError
from fun_with_ast.dynamic_matcher import GetDynamicMatcher


class JoinStrMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.JoinedStr([create_node.Str('fun-with-ast')])
        string = "f'fun-with-ast'"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchDoubleQuote(self):
        node = create_node.JoinedStr([create_node.Str('fun-with-ast')])
        string = "f\"fun-with-ast\""
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicnNMatchDoubleQuote(self):
        node = create_node.JoinedStr([create_node.Str('fun_with-ast')])
        string = "f\"fun-with-ast\""
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)

    def testBasicnNMatchEmpty(self):
        node = create_node.JoinedStr([create_node.Str('')])
        string = "f\"\""
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicFormatedValuey(self):
        node = create_node.JoinedStr([create_node.FormattedValue(create_node.Name('a'))])
        string = "f'{a}'"
#        matcher = GetDynamicMatcher(node)
#        matcher.Match(string)
#        matched_string = matcher.GetSource()
#        self.assertEqual(string, matched_string)