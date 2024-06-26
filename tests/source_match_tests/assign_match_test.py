import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class AssignMatcherTest(BaseTestUtils):

    def testBasicMatchAssignHexWithUpper(self):
        node = create_node.Assign('a', create_node.Num('0x1F')) # not implemented matching to 0x1f
        string = 'a=0x1F'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)
    def testBasicMatchAssignHexWithLower(self):
        node = create_node.Assign('a', create_node.Num('0x1f'))
        string = 'a=0x1f'
        self._assert_matched_source(node, string)


    def testBasicNoMatchAssignNone(self):
        node = create_node.Assign('a', create_node.CreateNone('None'))
        string = 'a = \t none # a is None'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._assert_matched_source(node, string)

    def testBasicMatchAssignString(self):
        node = create_node.Assign('a', create_node.Constant('1', "'"))
        string = "a='1'"
        self._assert_matched_source(node, string)

    def testBasicNoMatchAssignStringWithDoubleQuote(self):
        node = create_node.Assign('a', create_node.Constant('1', "'"))
        string = "a=\"1\""
        self._assert_matched_source(node, string)

    def testBasicMatchAssignStringWithDoubleQuote2(self):
        node = create_node.Assign('a', create_node.Constant('1', "\""))
        string = "a=\"1\""
        self._assert_matched_source(node, string)

    def testBasicMatchAssignString2(self):
        node = create_node.Assign('a', create_node.Constant('12', "\'"))
        string = "a='1''2'"
        self._assert_matched_source(node, string)



    def testBasicMatchAssign(self):
        node = create_node.Assign('a', create_node.Num('1'))
        string = 'a=1'
        self._assert_matched_source(node, string)

    def testBasicMatchAssignWithNL(self):
        node = create_node.Assign('a', create_node.Num('2'))
        string = 'a=2'
        self._assert_matched_source(node, string)


    def testBasicMatchAssignWithWSAndTab(self):
        node = create_node.Assign('a', create_node.Num('1'))
        string = 'a  =  1  \t'
        self._verify_match(node, string, trim_suffix_spaces=True)

    def testMatchMultiAssign(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('2'))
        string = 'a=b=1'
        matcher = GetDynamicMatcher(node)
        matched_string = matcher.GetSource()
        self.assertNotEqual(string, matched_string)

    def testNotMatchMultiAssign(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('1'))
        string = 'a=c=1'
        matcher = GetDynamicMatcher(node)
        matched_string = matcher.GetSource()
        self.assertNotEqual(string, matched_string)

    def testNotMatchMultiAssign2(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('1'))
        string = 'a=c=1\n'
        matcher = GetDynamicMatcher(node)
        matched_string = matcher.GetSource()
        self.assertNotEqual(string, matched_string)


    def testMatchMultiAssignWithWS(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('0o7654'))
        string = 'a\t=\t     b \t  =0o7654'
        self._assert_matched_source(node, string)

    def testMatchMultiAssignWithWSAndComment(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('1'))
        string = 'a\t=\t     b \t  =1 \t #comment'
        self._assert_matched_source(node, string)

    def testMatchMultiAssignNameWithWSAndComment(self):
        node = create_node.Assign(['a', 'b'], create_node.Name('c'))
        string = 'a\t=\t     b \t  =c \t #comment'
        self._assert_matched_source(node, string)

    def testMatchMultiAssignNameWithWSAndComment3(self):
        node = create_node.Assign(['a', 'b'], create_node.Name('c'))
        string = 'a\t=\t     b \t  =c \t #########'
        self._assert_matched_source(node, string)

    def testNotMatchMultiAssignWithWS(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('1'))
        string = 'a\t=\t     bb \t  =1 \t'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)

###############################################################################
    #test assign from source
###############################################################################
    def testAssignFromSource(self):
        string = 'a=1'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSource1(self):
        string = 'a=1\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSource1_1(self):
        string = 'a=1      \n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSource1_2(self):
        string = 'a=1\n     '
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testAssignFromSource1_3(self):
        string = 'a=1\n     '
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string, trim_suffix_spaces=True)


    def testAssignFromSource1_4(self):
        string = 'a=(1)\n     '
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)

    def testAssignFromSource2(self):
        string = "a='str'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSource3(self):
        string = "a=\"str'\""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSource4(self):
        string = "reduction_axes = (-1,)"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def _assert_matched_source(self, node, string):
        self._verify_match(node, string)

    def testAssignFromSourceList(self):
        string = "select = ['name', 'shares', 'price'],  # <-- See this line of AST unparse results"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    def testAssignFromSourceList2(self):
        string = "layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceList21(self):
        string = "layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1)]"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceList22(self):
        string = "layers = [nn.Conv2d(nc, ndf, kernel_size=4,), nn.LeakyReLU(0.2)]"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSourceList23(self):
        string = "layers = [nn.Conv2d(kernel_size=4), nn.LeakyReLU(0.2)]"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceList24(self):
        string = "layers = [nn.Conv2d(kernel_size=4,), nn.LeakyReLU(0.2)]"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceList25(self):
        string = """causal_mask = lax.dynamic_slice(
            self.causal_mask,
            (0, 0, mask_shift, 0),
            (1, 1, query_length, max_decoder_length),
        )"""
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string)

    def testAssignFromSourceList252(self):
        string = """causal_mask = lax.dynamic_slice(
            self.causal_mask,
            (0, 0, mask_shift, 0),
            (1, 1, query_length, max_decoder_length),
        )
        """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testAssignFromSourceList251(self):
        string = """causal_mask = lax.dynamic_slice(
            (0, 0),
            (1, 1),
        )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithJoinedStr1(self):
        string = """new_k = (
        *k[:name_idx],
        f"{k[name_idx][:-1]}_{i}",  
    )      """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithJoinedStr11(self):
        string = """new_k = (
        *k[:name_idx],
        f"{k[name_idx][:-1]}_{i}",
    )      
    """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testAssignFromSourceWithJoinedStr12(self):
        string = """new_k = (
        f"{k[name_idx][:-1]}_{i}",
    )
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testAssignFromSourceWithJoinedStr13(self):
        string = """new_k = (*k[:name_idx],           f"{k[name_idx][:-1]}_{i}",)      """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testAssignFromSourceWithDict(self):
        string = """model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
    input_ids,
    params,
    {"attention_mask": attention_mask, **model_kwargs_input},
)"""
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string)

    def testAssignFromSourceWithDict1(self):
        string = """model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
    input_ids,
    params,
    {"attention_mask": attention_mask, **model_kwargs_input},
)
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testAssignFromSourceWithDict2(self):
        string = """model_kwargs = a(
    input_ids,
    params,
    {"attention_mask": attention_mask, **model_kwargs_input},
)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithDict21(self):
        string = """model_kwargs = a(
    input_ids,
    params,
    {"attention_mask": attention_mask, **model_kwargs_input},
)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithDict3(self):
        string = """m = a(
    {"attention_mask": attention_mask, **model_kwargs_input},
)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithDict4(self):
        string = """m = a(
    {"attention_mask": aaa, **mmm},
)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithDict5(self):
        string = """m = a({"attention_mask": aaa, **mmm},)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSourceWithDict6(self):
        string = """m = {*mmm}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithDict7(self):
        string = """m = {**mmm}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
