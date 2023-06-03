from unittest import TestCase

from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class BaseTestUtils(TestCase):
    def _verify_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        result_from_matcher = matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source, 'matcher GetSource result does not equal to original string')
        self.assertEqual(result_from_matcher,string, 'matcher Match result does not equal to original string')
