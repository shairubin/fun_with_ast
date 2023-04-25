import unittest

import pytest
from fun_with_ast.exceptions_source_match import BadlySpecifiedTemplateError

from dynamic_matcher import GetDynamicMatcher
from fun_with_ast.create_node import SyntaxFreeLine

import create_node
import source_match


class SyntaxFreeLineMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = SyntaxFreeLine()
        string = '\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual('\n', matcher.GetSource())

    def testVeryShortMatch(self):
        node = SyntaxFreeLine(
            comment='', col_offset=4, comment_indent=0)
        string = '    #  \n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testCommentMatch(self):
        node = SyntaxFreeLine(
            comment='comment', col_offset=1, comment_indent=3)
        string = ' #   comment \n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_text = matcher.GetSource()
        self.assertEqual(string, matched_text)

    @pytest.mark.xfail(strict=True)
    def testIndentedCommentMatch(self):
        node = SyntaxFreeLine(
            comment='comment', col_offset=1, comment_indent=2)
        string = ' # \t comment \n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)

        self.assertEqual(string, matcher.GetSource())

    def testOffsetCommentMatch(self):
        node = SyntaxFreeLine(
            comment='comment', col_offset=2, comment_indent=2)
        string = '  #  comment   \n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testChangeComment(self):
        node = SyntaxFreeLine(
            comment='comment', col_offset=1, comment_indent=0)
        string = ' #comment\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        node.col_offset = 1
        node.comment_indent = 1
        node.comment = 'hello'
        self.assertEqual(' # hello\n', matcher.GetSource())

    def testNotCommentFails(self):
        node = SyntaxFreeLine(
            comment='comment', col_offset=0, comment_indent=0)
        string = 'comment\n'
        matcher = GetDynamicMatcher(node)
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.Match(string)
