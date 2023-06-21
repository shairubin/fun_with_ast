import unittest

import pytest
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.manipulate_node import create_node
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ConstantStrMatcherTest(BaseTestUtils):



    def testBasicMatchStr(self):
        node = create_node.Str('1')
        string = "'1'"
        self._verify_match(node, string)


    def testBasicMatchStrDoubelQ(self):
        node = create_node.Str("1")
        string = "\"1\""
        self._verify_match(node, string)
        # matcher = GetDynamicMatcher(node)
        # self._validate_match(matcher, string)

    def testBasicMatchEmpty(self):
        node = create_node.Str('')
        string = "''"
        self._verify_match(node, string)
        # matcher = GetDynamicMatcher(node)
        # self._validate_match(matcher, string)

    def testBasicMatchEmpty2(self):
        node = create_node.Str('')
        string = "\"\""
        self._verify_match(node, string)
        # matcher = GetDynamicMatcher(node)
        # self._validate_match(matcher, string)

    def testBasicMatchEmpty3(self):
        node = create_node.Str('')
        string = "\"\"''"
        self._verify_match(node, string)

        # matcher = GetDynamicMatcher(node)
        # self._validate_match(matcher, string)

    def testBasicMatchMultiPart(self):
        node = create_node.Str("'1''2'")
        string = "\"'1''2'\""
        self._verify_match(node, string)

        # matcher = GetDynamicMatcher(node)
        # self._validate_match(matcher, string)

    def testBasicMatchMultiPart2(self):
        node = create_node.Str('1''2')
        string = '\'1\'\'2\''
        self._verify_match(node, string)
        # matcher = GetDynamicMatcher(node)
        # self._validate_match(matcher, string)

    def testNoMatchMultiPart(self):
        node = create_node.Str("\"'1''2'\"")
        string = "\"'1''3'\""
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError) as e:
            match_result = matcher.do_match(string)
        #assert match_result != string


    def testBasicMatchConcatinatedString(self):
        node = create_node.Str('1''2')
        string = "'12'"
        self._verify_match(node, string)
        # matcher = GetDynamicMatcher(node)
        # self._validate_match(matcher, string)

    def testBasicMatchStrWithWS(self):
        node = create_node.Str('  1  ')
        string = "'  1  '"
        self._verify_match(node, string)
        # matcher = GetDynamicMatcher(node)
        # self._validate_match(matcher, string)

    def testBasicNoMatchStr(self):
        node = create_node.Str('1')
        string = "'2'"
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)

    def _validate_match(self, matcher, string):
        matcher.do_match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)
