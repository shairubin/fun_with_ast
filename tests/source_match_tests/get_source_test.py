import unittest

from fun_with_ast.dynamic_matcher import GetDynamicMatcher
from manipulate_node.create_node import GetNodeFromInput
from fun_with_ast.get_source import GetSource
from manipulate_node.if_manipulator import ManipulateIfNode


class GetSourceTest(unittest.TestCase):
    def testCall1(self):
        string = "logger.info('test string')\n"
        self._verify_source(string)

    def testCall2(self):
        string = 'logger.info(\'test string\')\n'
        self._verify_source(string)

    def _verify_source(self, string):
        log_node = GetNodeFromInput(string)
        source = GetSource(log_node, string)
        self.assertEqual(string, source)

    def testIf(self):
        string = 'if True:\n   a=1'
        log_node = GetNodeFromInput(string)
        GetSource(log_node, string)
        l1 = GetSource(log_node.body[0])
        assert l1 != '\n'
        self.assertEqual(string, GetSource(log_node))

    def testIf2(self):
        string = 'if True:\n   b.a(c)\n   a=1'
        log_node = GetNodeFromInput(string)
        source = GetSource(log_node, string)
        l1 = GetSource(log_node.body[0])
        assert l1 != '\n'
        self.assertEqual(string, source)

    def testCompose(self):
        string1 = 'if (c.d()):\n   a=1'
        if_node = GetNodeFromInput(string1)
        matcher1 = GetDynamicMatcher(if_node)
        matcher1.Match(string1)
        source1 = matcher1.GetSource()
        self.assertEqual(source1, string1)
        string2 = 'a.b()\n'
        call_node = GetNodeFromInput(string2)
        matcher2 = GetDynamicMatcher(call_node)
        matcher2.Match(string2)
        source2 = matcher2.GetSource()
        self.assertEqual(source2, string2)
        manipulator = ManipulateIfNode(if_node)
        manipulator.add_nodes_to_body([call_node],1)
        composed_source = GetSource(if_node, assume_no_indent=True)
        expected_source = string1 + '\n   ' + string2
        self.assertEqual(expected_source, composed_source)
