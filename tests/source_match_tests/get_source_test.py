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
        original_if_source = 'if (c.d()):\n   a=1'
        if_node = GetNodeFromInput(original_if_source)
        if_node_matcher = GetDynamicMatcher(if_node)
        if_node_matcher.Match(original_if_source)
        if_node_source = if_node_matcher.GetSource()
        self.assertEqual(if_node_source, original_if_source)
        injected_call_string = 'a.b()\n'
        call_node = GetNodeFromInput(injected_call_string)
        matcher2 = GetDynamicMatcher(call_node)
        matcher2.Match(injected_call_string)
        source2 = matcher2.GetSource()
        self.assertEqual(source2, injected_call_string)
        manipulator = ManipulateIfNode(if_node)
        manipulator.add_nodes_to_body([call_node],1)
        composed_source = GetSource(if_node, assume_no_indent=True)
        expected_source = original_if_source + '\n   ' + injected_call_string
        self.assertEqual(expected_source, composed_source)
