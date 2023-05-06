import unittest

import pytest
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

import create_node
from fun_with_ast.dynamic_matcher import GetDynamicMatcher


class ParenWrappedTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Name('a')
        string = '(a)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


    def testNewLineMatch(self):
        node = create_node.Name('a')
        string = '(\na\n)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_text = matcher.GetSource()
        self.assertEqual(string, matched_text)


    def testLeadingSpaces(self):
        node = create_node.Name('a')
        string = '  a'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_text = matcher.GetSource()
        self.assertEqual(string, matched_text)

    def testMatchTrailingTabs(self):
        node = create_node.Name('a')
        string = '(a  \t  )  \t '
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_text = matcher.GetSource()
        self.assertEqual(string, matched_text)

    @pytest.mark.xfail(strict=True)
    def testNoMatchLeadingTabs(self):
        node = create_node.Name('a')
        string = ' \t (a  \t  )  \t '
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_text = matcher.GetSource()
        self.assertNotEqual(string, matched_text)

    def testMatchLeadingTabs(self):
        node = create_node.Name('a')
        string = ' \t\n  a'
        matcher = GetDynamicMatcher(node)
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.Match(string)


    def testWithOperatorAndLineBreaks(self):
        node = create_node.Compare('a', '<', 'c')
        string = '(a < \n c\n)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testWithOperatorAndLineBreaksAndTabs(self):
        node = create_node.Compare('a', '<', 'c')
        string = ' (a < \n\t  c\n)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


    def testWithTuple(self):
        node = create_node.Call('c', args=[create_node.Name('d'),
                                           create_node.Tuple(['a', 'b'])])
        string = ' c(d, (a, b))'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
        string = ' c (d, (a, b))'
        matcher = GetDynamicMatcher(node)
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.Match(string)

