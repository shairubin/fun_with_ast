import unittest

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.get_source import GetSource
from tests.source_match_tests.base_test_utils import BaseTestUtils


class GetSourceTest(BaseTestUtils):
    def testCall1(self):
        string = "logger.info(\"test string\")\n"
        self._verify_source(string)

    def testEpmty(self):
        string = ""
        self._verify_source(string)

    def testCall2(self):
        string = 'logger.info(\'test string\')\n'
        self._verify_source(string)

    def testImport(self):
        string = 'import a\nimport b\n'
        self._verify_source(string, get_module=True)

    def testImportWithComment(self):
        string = 'import a # comment 1\nimport b #comment 2\n'
        self._verify_source(string, get_module=True)

    def testImportWithComment2(self):
        string = '#comment 0\nimport a # comment 1\nimport b #comment 2\n'
        self._verify_source(string, get_module=True)

    @pytest.mark.skip('Not implemented yet')
    def testImportWithComment3(self):
        string = '#comment 0\nimport a # comment 1\nimport b #comment 2\n# comment end'
        self._verify_source(string, default_quote='\"', get_module=True)

    #@pytest.mark.xfail(reason='Not implemented yet', raises=AssertionError)
    def testCall3(self):
        string = 'logger.info(\'test string\')\n'
        self._verify_source(string)

    def testForAndIf(self):
        string = """for i in range(1, 15):\n print('fun with ast')\n pass"""
        self._verify_source(string, get_module=True)

    def testIf(self):
        string = 'if True:\n   a=1'
        self._verify_match_and_no_new_line(string)

    def _verify_match_and_no_new_line(self, string):
        log_node = GetNodeFromInput(string)
        GetSource(log_node, string)
        l1 = GetSource(log_node.body[0])
        assert l1 != '\n'
        self.assertEqual(string, GetSource(log_node))


    def testIf2(self):
        string = 'if True:\n   b.a(c)\n   a=1'
        self._verify_match_and_no_new_line(string)

    def testIf3(self):
        string = 'if (x == 0xff):\n   b.a(c)\n   a=1'
        self._verify_match_and_no_new_line(string)

    def testIAssign1(self):
        string = "a='fun_with_east'"
        log_node = GetNodeFromInput(string)
        source = GetSource(log_node, string)
        self.assertEqual(string, source)

    def testIAssign3(self):
        string = "a=1"
        log_node = GetNodeFromInput(string)
        source = GetSource(log_node, string)
        self.assertEqual(string, source)
    def testIAssign4(self):
        string = "a=0xff"
        log_node = GetNodeFromInput(string)
        source = GetSource(log_node, string)
        self.assertEqual(string, source)

    def testConstant(self):
        string = "\"fun_with_east\"\n"
        self._verify_source(string)

    @pytest.mark.xfail()
    def testIAssign2(self):
        string1 = "a='fun_with_east'"
        n1 = GetNodeFromInput(string1)
        source1 = GetSource(n1, string1)
        self.assertEqual(string1, source1)
        string2 = "b='more_fun_with_east'"
        n2 = GetNodeFromInput(string2)
        source2 = GetSource(n2, string2)
        self.assertEqual(string2, source2)
        if_string = 'if True:\n   pass\n   a=1'
        if_node = GetNodeFromInput(if_string)
        if_source = GetSource(if_node, if_string)
        self.assertEqual(if_string, if_source)
        if_node.body[0] = n1
        if_node.body[1] = n2
        if_source = GetSource(if_node)
        if_modified_string = if_string.replace('  pass',string1)
        if_modified_string = if_modified_string.replace('  a=1', string2)
        self.assertEqual(if_modified_string, if_source)

    def _verify_source(self, string, get_module=False):
        node = GetNodeFromInput(string, 0, get_module=get_module)
        source = GetSource(node, string)
        self.assertEqual(string, source)
        node = GetNodeFromInput(string,0,get_module)
        if not get_module:
            source = GetSource(node, assume_no_indent=True)
            #if default_quote == '\"':
            #    string = string.replace("\'", '\"')
            self.assertEqual(string, source)
