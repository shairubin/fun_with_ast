import unittest

import pytest
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

import create_node
from fun_with_ast.dynamic_matcher import GetDynamicMatcher


class NameMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Name('foobar')
        string = 'foobar'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithWS(self):
        node = create_node.Name('foobar')
        string = ' \t  foobar \t'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    @pytest.mark.xfail(strict=True)
    def testBasicMatchWithWSAndComment(self):
        node = create_node.Name('foobar')
        string = ' \t  foobar \t #comment'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchOnlyComment(self):
        node = create_node.Name('foobar')
        string = ' \t  foobar#comment'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchOnlyComment(self):
        node = create_node.Name('foobar')
        string = ' \t #comment  foobar'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)

    def testIdChange(self):
        node = create_node.Name('foobar')
        string = 'foobar'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        node.id = 'hello'
        self.assertEqual('hello', matcher.GetSource())


    def testBasicMatch2(self):
        node = create_node.Name('a')
        string = 'a'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithWS(self):
        node = create_node.Name('a')
        string = 'a '
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    @pytest.mark.xfail(strict=True)
    def testMatchWithComment(self):
        node = create_node.Name('a')
        string = 'a # comment'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testLeadingSpaces(self):
        node = create_node.Name('a')
        string = '  a'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_text = matcher.GetSource()
        self.assertEqual(string, matched_text)
        string = ' \t  a'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_text = matcher.GetSource()
        self.assertEqual(string, matched_text)
        string = ' \t\n  a'
        matcher = GetDynamicMatcher(node)
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.Match(string)

