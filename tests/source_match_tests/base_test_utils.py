import ast
from unittest import TestCase

from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.source_matchers.reset_match import ResetMatch


#class ResetMatcher:
#    pass


class BaseTestUtils(TestCase):
    # get_source_after_reset: if True, then we will get source after reset_match. it will not be the same
    # as the original source in some specific cases. For example, 'a=1e-6' MUST be matched before calling to GetSource.
    # Otherwise, the result would be a=1e-06
    def _verify_match(self, node, string, get_source_after_reset=True, trim_suffix_spaces=False):
        if trim_suffix_spaces:
            assert not isinstance(node,ast.Module)

        trimmed_string = string
        while trimmed_string and trimmed_string[-1] in [' ', '\t']:
            trimmed_string = trimmed_string[:-1]
        matcher = GetDynamicMatcher(node)
        assert hasattr(node, 'node_matcher')
        result_from_matcher = matcher.do_match(string)
        if trim_suffix_spaces:
            self.assertEqual(result_from_matcher, trimmed_string, 'matcher do_match does not equal to original string')
        else:
            self.assertEqual(result_from_matcher, string, 'matcher do_match does not equal to original string')
        matcher_source = matcher.GetSource()
        node_matcher_source = node.node_matcher.GetSource()
        self.assertEqual(matcher_source, node_matcher_source,
                         'matcher GetSource result does not equal to node_matcher GetSource result')
        self.assertEqual(result_from_matcher, node_matcher_source,
                         'result_from_matcher does not equal to node_matcher GetSource result')
        reseter = ResetMatch(node)
        reseter.reset_match()
        if get_source_after_reset:
            matcher_source_after_reset = matcher.GetSource()
            self.assertEqual(matcher_source_after_reset, node_matcher_source,
                             'matcher_source_after_reset does not equal to node_matcher GetSource result')
