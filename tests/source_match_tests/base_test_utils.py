from unittest import TestCase

from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.source_matchers.reset_match import ResetMatch


class ResetMatcher:
    pass


class BaseTestUtils(TestCase):
    def _verify_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        assert hasattr(node, 'node_matcher')
        result_from_matcher = matcher.do_match(string)
        matcher_source = matcher.GetSource()
        node_matcher_source = node.node_matcher.GetSource()
        self.assertEqual(string, matcher_source, 'matcher GetSource result does not equal to original string')
        self.assertEqual(result_from_matcher,string, 'matcher Match result does not equal to original string')
        self.assertEqual(matcher_source, node_matcher_source, 'matcher GetSource result does not equal to node_matcher GetSource result')
        self.assertEqual(result_from_matcher, node_matcher_source, 'result_from_matcher does not equal to node_matcher GetSource result')
        reseter = ResetMatch(node)
        reseter.reset_match()

