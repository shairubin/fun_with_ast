import unittest

import pytest

import create_node
import source_match


class ParenWrappedTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Name('a')
        string = '(a)'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


    def testNewLineMatch(self):
        node = create_node.Name('a')
        string = '(\na\n)'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        matched_text = matcher.GetSource()
        self.assertEqual(string, matched_text)


    def testLeadingSpaces(self):
        node = create_node.Name('a')
        string = '  a'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        matched_text = matcher.GetSource()
        self.assertEqual(string, matched_text)

    def testMatchTrailingTabs(self):
        node = create_node.Name('a')
        string = '(a  \t  )  \t '
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        matched_text = matcher.GetSource()
        self.assertEqual(string, matched_text)

    @pytest.mark.xfail(strict=True)
    def testNoMatchLeadingTabs(self):
        node = create_node.Name('a')
        string = ' \t (a  \t  )  \t '
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        matched_text = matcher.GetSource()
        self.assertNotEqual(string, matched_text)

    def testMatchLeadingTabs(self):
        node = create_node.Name('a')
        string = ' \t\n  a'
        matcher = source_match.GetMatcher(node)
        with self.assertRaises(source_match.BadlySpecifiedTemplateError):
            matcher.Match(string)


    def testWithOperatorAndLineBreaks(self):
        node = create_node.Compare('a', '<', 'c')
        string = '(a < \n c\n)'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testWithOperatorAndLineBreaksAndTabs(self):
        node = create_node.Compare('a', '<', 'c')
        string = ' (a < \n\t  c\n)'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


    def testWithTuple(self):
        node = create_node.Call('c', args=[create_node.Name('d'),
                                           create_node.Tuple(['a', 'b'])])
        string = ' c(d, (a, b))'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
        string = ' c (d, (a, b))'
        matcher = source_match.GetMatcher(node)
        with self.assertRaises(source_match.BadlySpecifiedTemplateError):
            matcher.Match(string)

