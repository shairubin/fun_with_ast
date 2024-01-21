import pytest

from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class SubscriptMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Subscript('a', 1)
        string = 'a[1]'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual('a[1]', matcher.GetSource())

    def testAllPartsMatch(self):
        node = create_node.Subscript('a', 1, 2, 3)
        string = 'a[1:2:3]'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual('a[1:2:3]', matcher.GetSource())

    def testSeparatedWithStrings(self):
        node = create_node.Subscript('a', 1, 2, 3)
        string = 'a [ 1 : 2 : 3 ]'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual('a [ 1 : 2 : 3 ]', matcher.GetSource())

    def testSubscriptModule7Partial(self):
        string =  """f"k[a][:-1]" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testSubscriptModule7Partial2(self):
        string =  """f"{k[1:]}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testSubscriptModule7Partial3(self):
        string =  """k[1:] """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testSubscriptModule7Partial5(self):
        string =  'k[:1]'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testSubscriptSlices(self):
        string = 'a[:, :, :query_length, :key_length]'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

###################################################################
###################  GetSource tests NOT SUPPORTED ################
###################################################################
# GetSource Without getting string to match cannot support all scenarios
#  because it cannot know the parts of the slices and therefore assumes
#  they are all  ':'
    def testSubscriptGetSource(self):
        string = 'k[1]'
        node = GetNodeFromInput(string)
        source = GetSource(node.value)
        assert source == string

    @pytest.mark.skip(reason="GetSource Without getting string to match cannot support all scenarios")
    def testSubscriptGetSource1(self):
        string = 'k[1:]'
        node = GetNodeFromInput(string)
        source = GetSource(node.value)
        assert source == string
    @pytest.mark.skip(reason="GetSource Without getting string to match cannot support all scenarios")
    def testSubscriptGetSource2(self):
        string = 'k[1:2]'
        node = GetNodeFromInput(string)
        source = GetSource(node.value)
        assert source == string
    @pytest.mark.skip(reason="GetSource Without getting string to match cannot support all scenarios")
    def testSubscriptGetSource3(self):
        string = 'k[:2]'
        node = GetNodeFromInput(string)
        source = GetSource(node.value)
        assert source == string