import unittest

import pytest
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.manipulate_node.create_node import SyntaxFreeLine
from tests.source_match_tests.base_test_utils import BaseTestUtils


class SyntaxFreeLineMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = SyntaxFreeLine()
        string = '\n'
        self._verify_match(node, string)

    def testBasicMatch2(self):
        node = SyntaxFreeLine()
        string = '# comment\n'
        self._verify_match(node, string)

    def testVeryShortMatch(self):
        node = SyntaxFreeLine(
            comment='', col_offset=4, comment_indent=0)
        string = '    #  \n'
        self._verify_match(node, string)

    def testCommentMatch(self):
        node = SyntaxFreeLine(
            comment='comment', col_offset=1, comment_indent=3)
        string = ' #   comment \n'
        self._verify_match(node, string)


    @pytest.mark.skip(reason="Not Implemented Yet")
    def testIndentedCommentMatch(self):
        node = SyntaxFreeLine(
            comment='comment', col_offset=1, comment_indent=2)
        string = ' # \t comment \n'
        self._verify_match(node, string)


    def testOffsetCommentMatch(self):
        node = SyntaxFreeLine(
            comment='comment', col_offset=2, comment_indent=2)
        string = '  #  comment   \n'
        self._verify_match(node, string)



    def testNotCommentFails(self):
        node = SyntaxFreeLine(
            comment='comment', col_offset=0, comment_indent=0)
        string = 'comment\n'
        matcher = GetDynamicMatcher(node)
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.do_match(string)
