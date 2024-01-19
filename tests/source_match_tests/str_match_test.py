import unittest

import pytest

from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.manipulate_node import create_node
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ConstantStrMatcherTest(BaseTestUtils):



    def testBasicMatchStr(self):
        node = create_node.Constant('1', "'")
        string = "'1'"
        self._verify_match(node, string)


    def testBasicMatchStrDoubelQ(self):
        node = create_node.Constant("1", "\"")
        string = "\"1\""
        self._verify_match(node, string)


    def testBasicMatchEmpty(self):
        node = create_node.Constant('', "'")
        string = "''"
        self._verify_match(node, string)

    def testBasicMatchEmptyDoubleQ(self):
        node = create_node.Constant('', "\"")
        string = "\"\""
        self._verify_match(node, string)





    def testBasicMatchConcatinatedString(self):
        node = create_node.Constant('1''2', "'")
        string = "'12'"
        self._verify_match(node, string)

    def testBasicMatchStrWithWS(self):
        node = create_node.Constant('  1  ', "'")
        string = "'  1  '"
        self._verify_match(node, string)

    def testBasicMatchStrWithNL(self):
        node = create_node.Constant('  1  ', "'")
        string = "'  1  '"
        self._verify_match(node, string)

    def testBasicNoMatchStr(self):
        node = create_node.Constant('1', "'")
        string = "'2'"
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)
########################################################################
# get node tests
    def testSimple(self):
        string = "'X'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithNLString(self):
        string = "\"\\n\""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testStringWithNLString2(self):
        string = "'\\n'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithNLString3(self):
        string = "'abc\\n'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithNLString4(self):
        string = "'abc\ndef'"
        with pytest.raises(SyntaxError):
            node = GetNodeFromInput(string)
    def testStringWithNLString5(self):
        string = "'abc\\ndef'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringMultipart(self):
        string = "'abc''def'"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testStringMultipart2(self):
        string = "('abc'\n'def')"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testStringMultipart4(self):
        string = "'abc'\t'def'"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testStringMultipart5(self):
        string = "'abc'\n'def'"
        node = GetNodeFromInput(string, get_module=True)
        with pytest.raises(BadlySpecifiedTemplateError, match='.*two consecutive strings with new-line seperator between them.*'):
            self._verify_match(node, string)

    def testStringMultipart3(self):
        string = "'abc''def'"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testStringSinglePart(self):
        string = "'abcdef'"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testStringSingleQuoteAndBackslash(self):
        string = """'Module must support a \\'device\\' arg to skip initialization'"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringSingleQuoteAndBackslash(self):
        string = "'A\\'B\\'C'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testNativeGetSource(self):
        string = "'abc' "
        node = GetNodeFromInput(string)
        source = GetSource(node.value)
        with pytest.raises(AssertionError):
            assert source == string, "The space is missing since this is NOT part of an expression"

    @pytest.mark.skip(reason="issue #196")
    def testNativeGetSource2(self):
        string = """"abc" """
        node = GetNodeFromInput(string)
        source = GetSource(node.value)
        assert source == string
