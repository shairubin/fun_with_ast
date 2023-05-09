import unittest

import pytest

from fun_with_ast.dynamic_matcher import GetDynamicMatcher
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

from manipulate_node import create_node


class ConstantNumMatcherTest(unittest.TestCase):

    def testBasicMatchNum(self):
        node = create_node.Num('1')
        string = '1'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)


    def testBasicMatchWithPlusSign(self):
        node = create_node.Num('1')
        string = '+1'
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
        node = create_node.Num('1')
        string = '((   1   )    ) '
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testNoMatchMultiParansAndWS(self):
        node = create_node.Num('1')
        string = '((   1   )     '
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertNotEqual(string, matcher.GetSource())

    def testLargeNumberMatch(self):
        node = create_node.Num('1234567890987654321')
        string = '1234567890987654321'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicNoMatchNum(self):
        node = create_node.Num('2')
        string = '1'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)
        string = '2'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


    def testMatchBool(self):
        node = create_node.Bool(False)
        string = 'False'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchBoolParans(self):
        node = create_node.Bool(False)
        string = '(False)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

class ConstantStrMatcherTest(unittest.TestCase):



    def testBasicMatchStr(self):
        node = create_node.Str('1')
        string = "'1'"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicMatchStrDoubelQ(self):
        node = create_node.Str("1")
        string = "\"1\""
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicMatchEmpty(self):
        node = create_node.Str('')
        string = "''"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicMatchEmpty2(self):
        node = create_node.Str('')
        string = "\"\""
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicMatchEmpty3(self):
        node = create_node.Str('')
        string = "\"\"''"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicMatchMultiPart(self):
        node = create_node.Str("'1''2'")
        string = "\"'1''2'\""
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicMatchMultiPart2(self):
        node = create_node.Str('1''2')
        string = '\'1\'\'2\''
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)


    def testNoMatchMultiPart(self):
        node = create_node.Str("\"'1''2'\"")
        string = "\"'1''3'\""
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)


    def testBasicMatchConcatinatedString(self):
        node = create_node.Str('1''2')
        string = "'12'"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicMatchStrWithWS(self):
        node = create_node.Str('  1  ')
        string = "'  1  '"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicNoMatchStr(self):
        node = create_node.Str('1')
        string = "'2'"
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)

