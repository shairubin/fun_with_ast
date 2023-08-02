import unittest

import pytest
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ParenWrappedTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Name('a')
        string = '(a)'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())


    def testNewLineMatch(self):
        node = create_node.Name('a')
        string = '(\na\n)'
        self._assert_match(node, string)

    def testLeadingSpaces(self):
        node = create_node.Name('a')
        string = '  a'
        self._assert_match(node, string)

    def _assert_match(self, node, string):
        #matcher = GetDynamicMatcher(node)
        #matcher.do_match(string)
        #matched_text = matcher.GetSource()
        #self.assertEqual(string, matched_text)
        self._verify_match(node, string)

    def testMatchTrailingTabs(self):
        node = create_node.Name('a')
        string = '(a  \t  )  \t '
        self._assert_match(node, string)

    def testNoMatchLeadingTabs(self):
        node = create_node.Name('a')
        string = ' \t (a  \t  )  \t '
        self._assert_match(node, string)

    def testMatchLeadingTabs(self):
        node = create_node.Name('a')
        string = ' \t\n  a'
        matcher = GetDynamicMatcher(node)
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.do_match(string)


    def testWithOperatorAndLineBreaks(self):
        node = create_node.Compare('a', '<', 'c')
        string = '(a < \n c\n)'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testWithOperatorAndLineBreaksAndTabs(self):
        node = create_node.Compare('a', '<', 'c')
        string = ' (a < \n\t  c\n)'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())


    def testWithTuple(self):
        node = create_node.Call('c', args=[create_node.Name('d'),
                                           create_node.Tuple(['a', 'b'])])
        string = ' c(d, (a, b))'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())
        string = ' c (d, (a, b))'
        matcher = GetDynamicMatcher(node)
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.do_match(string)

