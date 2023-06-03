import unittest

import pytest

from fun_with_ast.manipulate_node.create_node import GetNodeFromInput
from fun_with_ast.get_source import GetSource


class GetSourceTest(unittest.TestCase):
    def testCall1(self):
        string = "logger.info(\"test string\")\n"
        self._verify_source(string, default_quote='\'')

    def testCall2(self):
        string = 'logger.info(\'test string\')\n'
        self._verify_source(string,default_quote='\"')

    @pytest.mark.xfail(reason='Not implemented yet', raises=AssertionError)
    def testCall3(self):
        string = 'logger.info(\'test string\')\n'
        self._verify_source(string,default_quote='\'')

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
        string = 'if x%2 == 0xff:\n   b.a(c)\n   a=1'
        self._verify_match_and_no_new_line(string)

    def testIAssign1(self):
        string = "a='fun_with_east'"
        log_node = GetNodeFromInput(string)
        source = GetSource(log_node, string)
        self.assertEqual(string, source)

    def testConstant(self):
        string = "\"fun_with_east\"\n"
        self._verify_source(string, default_quote='\"')

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

    def _verify_source(self, string, default_quote):
        log_node = GetNodeFromInput(string)
        source = GetSource(log_node, string)
        self.assertEqual(string, source)
        log_node = GetNodeFromInput(string)
        source = GetSource(log_node, assume_no_indent=True)
        if default_quote == '\"':
            string = string.replace("\'", '\"')
        self.assertEqual(string, source)
