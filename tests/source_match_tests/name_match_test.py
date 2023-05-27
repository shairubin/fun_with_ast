import unittest

import pytest
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class NameMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Name('foobar')
        string = 'foobar'
        self._assert_match(node, string)


    def testBasicMatchWithWS(self):
        node = create_node.Name('foobar')
        string = ' \t  foobar \t'
        self._assert_match(node, string)
    #pytest.mark.skip(reason="Not implemented yet")
    def testBasicMatchWithWSAndComment(self):
        node = create_node.Name('foobar')
        string = ' \t  foobar \t #comment'
        self._assert_match(node, string)

    def testBasicMatchOnlyComment(self):
        node = create_node.Name('foobar')
        string = ' \t  foobar#comment'
        self._assert_match(node, string)

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
        self._assert_match(node, string)

    def testMatchWithWS(self):
        node = create_node.Name('a')
        string = 'a '
        self._assert_match(node, string)

    #@pytest.mark.skip(reason="Not Implemented Yet")
    def testMatchWithComment(self):
        node = create_node.Name('a')
        string = 'a # comment'
        self._assert_match(node, string)
#        matcher = GetDynamicMatcher(node)
#        matcher.Match(string)
#        matched_string = matcher.GetSource()
#        self.assertEqual(string, matched_string)

    def testLeadingSpaces(self):
        node = create_node.Name('a')
        string = '  a'
        self._assert_match(node, string)
        # matcher = GetDynamicMatcher(node)
        # matcher.Match(string)
        # matched_text = matcher.GetSource()
        # self.assertEqual(string, matched_text)
        string = ' \t  a'
        self._assert_match(node, string)
        # matcher = GetDynamicMatcher(node)
        # matcher.Match(string)
        # matched_text = matcher.GetSource()
        # self.assertEqual(string, matched_text)
        string = ' \t\n  a'
        matcher = GetDynamicMatcher(node)
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.Match(string)

    def _assert_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_source = matcher.GetSource()
        self.assertEqual(string, matched_source)