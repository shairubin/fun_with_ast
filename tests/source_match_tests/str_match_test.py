import unittest

import pytest
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.manipulate_node import create_node



class ConstantStrMatcherTest(unittest.TestCase):



    def testBasicMatchStr(self):
        node = create_node.Str('1')
        string = "'1'"
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)


    def testBasicMatchStrDoubelQ(self):
        node = create_node.Str("1")
        string = "\"1\""
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testBasicMatchEmpty(self):
        node = create_node.Str('')
        string = "''"
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testBasicMatchEmpty2(self):
        node = create_node.Str('')
        string = "\"\""
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testBasicMatchEmpty3(self):
        node = create_node.Str('')
        string = "\"\"''"
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testBasicMatchMultiPart(self):
        node = create_node.Str("'1''2'")
        string = "\"'1''2'\""
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testBasicMatchMultiPart2(self):
        node = create_node.Str('1''2')
        string = '\'1\'\'2\''
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testNoMatchMultiPart(self):
        node = create_node.Str("\"'1''2'\"")
        string = "\"'1''3'\""
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError) as e:
            matcher.Match(string)


    def testBasicMatchConcatinatedString(self):
        node = create_node.Str('1''2')
        string = "'12'"
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testBasicMatchStrWithWS(self):
        node = create_node.Str('  1  ')
        string = "'  1  '"
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testBasicNoMatchStr(self):
        node = create_node.Str('1')
        string = "'2'"
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)

    def _validate_match(self, matcher, string):
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)
