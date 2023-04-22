import unittest

from fun_with_ast import source_match

from fun_with_ast.create_node import GetNodeFromInput


class GetSourceTest(unittest.TestCase):
    def testCall1(self):
        string = "logger.info('test string')"
        log_node = GetNodeFromInput(string)
        source_match.GetSource(log_node.value, string)
        self.assertEqual(string, source_match.GetSource(log_node.value))

    def testCall2(self):
        string = 'logger.info(\'test string\')'
        log_node = GetNodeFromInput(string)
        source_match.GetSource(log_node.value, string)
        self.assertEqual(string, source_match.GetSource(log_node.value))
    def testIf(self):
        string = 'if True:\n   a=1'
        log_node = GetNodeFromInput(string)
        source_match.GetSource(log_node, string)
        l1 = source_match.GetSource(log_node.body[0])
        assert l1 != '\n'
        self.assertEqual(string, source_match.GetSource(log_node))
