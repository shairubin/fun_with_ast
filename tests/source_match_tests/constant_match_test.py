import unittest

import pytest

from fun_with_ast.manipulate_node.create_node import GetNodeFromInput
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

from fun_with_ast.manipulate_node import create_node
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ConstantNumMatcherTest(BaseTestUtils):


    def testBasicMatchNum(self):
        node = create_node.Num('1')
        string = '1'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicMatchNumBinary(self):
        node = create_node.Num('0b0')
        string = '0b0'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicMatchNumWithError(self):
        node = create_node.Num('1')
        string = '1:#Comment'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertNotEqual(string, matched_string)


    # def testBasicMatchWithPlusSign(self):
    #     node = create_node.Num('1')
    #     string = '+1'
    #     matcher = GetDynamicMatcher(node)
    #     matcher.Match(string)
    #     matched_string = matcher.GetSource()
    #     self.assertEqual(string, matched_string)
    #
    # def testBasicMatchWithPlusSign2(self):
    #     node = create_node.Num('1')
    #     string = '(+1)'
    #     matcher = GetDynamicMatcher(node)
    #     matcher.Match(string)
    #     matched_string = matcher.GetSource()
    #     self.assertEqual(string, matched_string)
    @pytest.mark.skip('Not implemented yet')
    def testBasicMatchWithPlusSign3(self):
        node = create_node.Num('1')
        string = '(    (    +1   )   ) # comment'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)


    def testBasicMatchWithMinusSign(self):
        node = create_node.Num('-1')
        string = '  -1   \t'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicMatchWithdWS(self):
        node = create_node.Num('1')
        string = '   1   '
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testMatchWSWithComment(self):
        node = create_node.Num('1')
        string = '   1   #comment'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testWithParans(self):
        node = create_node.Num('1')
        string = '(1)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testWithParansAndWS(self):
        node = create_node.Num('1')
        string = '(   1   )     '
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testWithMultiParansAndWS(self):
        node = create_node.Num('-1')
        string = '((   -1   )    ) '
        self._verify_match(node, string)


    def testNoMatchMultiParansAndWS(self):
        node = create_node.Num('1')
        string = '((   1   )     '
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)
    def testNoMatchMultiParansAndWS2(self):
        node = create_node.Num('1')
        string = '(   1   )     )'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)

    def testLargeNumberMatch(self):
        node = create_node.Num('1234567890987654321')
        string = '1234567890987654321'
        self._verify_match(node, string)

    def testLargeNumberMatchHex(self):
        node = create_node.Num('0xffaa')
        string = '\t0xffaa\t #comment'
        self._verify_match(node, string)

    def testLargeNumberMatchHexNoMatch(self):
        node = create_node.Num('0xffab')
        string = '\t0xffaa\t #comment'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)

    def testBasicNoMatchNum(self):
        node = create_node.Num('2')
        string = '1'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)
        string = '2'
        self._verify_match(node, string)

    def testMatchBool(self):
        node = create_node.Bool(False)
        string = 'False'
        self._verify_match(node, string)

    def testMatchBoolParans(self):
        node = create_node.Bool(False)
        string = '(False)'
        self._verify_match(node, string)


