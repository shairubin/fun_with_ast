import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput, FailedToCreateNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class TupleTest(BaseTestUtils):

    def testBasicTuple(self):
        node = create_node.Tuple(['a', 'b'])
        string = '(a,b)'
        self._assert_match(node, string)

    def testBasicTupleNone(self):
        node = create_node.Tuple(['a', 'None'])
        string = '(a,None)'
        self._assert_match(node, string)


    def testBasicTupleNoParans(self):
        node = create_node.Tuple(['a', 'b'])
        string = 'a,b'
        self._assert_match(node, string)

    def testBasicTupleNoParansComment(self):
        node = create_node.Tuple(['a', 'b'])
        string = '\t a,\t\tb \t #comment'
        self._assert_match(node, string)

    def testBasicTupleNoIllegal(self):
        node = create_node.Tuple(['a', 'b'])
        string = '(\t a,\t\tb \t #comment'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)

    def testBasicSingleTuple(self):
        node = create_node.Tuple(['a'])
        string = '(\t   a, \t)'
        self._assert_match(node, string)
    def testBasicSingleTuple3(self):
        node = create_node.Tuple(['a'])
        string = '(a,)'
        self._assert_match(node, string)

    def testBasicSingleTuple4(self):
        node = create_node.Tuple(['a'])
        string = 'a,'
        self._assert_match(node, string)

    def testBasicSingleTuple2(self):
        node = create_node.Tuple(['a'])
        string = '(\t   a \t)'
        self._assert_match(node, string)

    def testTupleWithCommentAndWS2(self):
        node = create_node.Tuple(['a', 'b'])
        string = ' (\t   a, b \t)#comment'
        self._assert_match(node, string)

    def testTupleWithCommentAndWSAndConst(self):
        node = create_node.Tuple(['a', 1])
        string = ' (\t   a\t, 1 \t) \t #comment'
        self._assert_match(node, string)
    ###############################################################
    #################   Get Node From Input   ####################
    ###############################################################
    def testCreateNodeFromInput(self):
        string = '(\t   a\t, 1 \t) \t #comment'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput1(self):
        string = '(a, 1,) \t #comment'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput11(self):
        string = '(a, 1)  #comment'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInput2(self):
        string = '(a,)'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput3(self):
        string = '(a,  \t b,   )'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput4(self):
        string = 'a,  \t b,   '
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput5(self):
        string = '-1,'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput6(self):
        string = '(-1,)'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput7(self):
        string = '(1),'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInput7(self):
        string = '((1),),'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL(self):
        string = '((1,2), (3,4))'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL2(self):
        string = '((1,2), (3,4),)'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInputTupleWithEOL21(self):
        string = '((1,2), (3,4),)\n'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL22(self):
        string = '((1,2), (3,4),)\n     '
        node =GetNodeFromInput(string, get_module=True)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL23(self):
        string = '(1,2)\n     '
        node =GetNodeFromInput(string, get_module=True)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL24(self):
        string = '(1,2)\n     '
        node =GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string, trim_suffix_spaces=True)

    def testCreateNodeFromInputTupleWithEOL25(self):
        string = '(1,2)\n     '
        node =GetNodeFromInput(string,  get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)

    def testCreateNodeFromInputTupleWithEOL26(self):
        string = '(1,2)\n     '
        node =GetNodeFromInput(string,  get_module=True)
        with pytest.raises(AssertionError): # 'Module SHOULD catch training spaces'
            self._verify_match(node, string, trim_suffix_spaces=True)

    def testCreateNodeFromInputTupleWithEOL3(self):
        string = '((1,2),\n (3,4))'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL31(self):
        string = '(1,\n 2)'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTrailingComma(self):
        string = '(1,2),'
        node = GetNodeFromInput(string)
        self._assert_match(node, string)
    #while the below test passes the implementation is not correct in hamdling the commas
    def testCreateNodeFromInputTrailingComma2(self):
        string = '((1,2),(3,4),)'
        node = GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithJoinedStr(self):
        string = """(
        *k[:name_idx],
        f"{k[name_idx][:-1]}_{i}",
    )
"""
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithJoinedStr2(self):
        string = """(
        a,
        f"x",
    )
"""
        node = GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithMultiLineComments(self):
        string = """(
        a, #comment 1
        f, #comment 2
    )
"""
        node = GetNodeFromInput(string)
        self._assert_match(node, string)

    def _assert_match(self, node, string):
        self._verify_match(node, string)
    def testAssignFromSourceWithTuppleAsValue(self):
        string = """{"attention_mask": aaa, **mmm}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithTuppleAsValue2(self):
        string = """{"attention_mask": aaa}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithTuppleAsValue3(self):
        string = """{"attention_mask": (a,b)}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithTuppleAsValue31(self):
        string = """{"attention_mask": a,b}"""
        with pytest.raises(FailedToCreateNodeFromInput):
            node = GetNodeFromInput(string)

    def testAssignFromSourceWithTuppleAsValue4(self):
        string = """{"attention_mask": (a, **m)}"""
        with pytest.raises(FailedToCreateNodeFromInput):
            node = GetNodeFromInput(string)

    def testAssignFromSourceWithTuppleAsValue41(self):
        string = """{"attention_mask": a, **m}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithJstr(self):
        string = """(
            f"-DPYTHON_EXECUTABLE:FILEPATH={sys.executable}",
            f"-DPYTHON_INCLUDE_DIR={sysconfig.get_path('include')}",
        )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testAssignFromSourceWithJstr2(self):
        string = """(f"-DPYTHON_EXECUTABLE:FILEPATH={sys.executable}",f"-DPYTHON_INCLUDE_DIR={sysconfig.get_path('include')}",)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceMultiLine(self):
        string = """(
                        "Old",
                        "cu",
                    )
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceMultiLine2(self):
        string = """("Old",   
    "cu")"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceMultiLine2_1(self):
        string = """("Old"     ,
    "cu")"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceMultiLine3(self):
        string = """("Old",
"cu")"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSourceMultiLine4(self):
        string = """("Old"
,"cu")"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceMultiLine4_1(self):
        string = """("Old"
    ,   "cu")"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceMultiLine4_2(self):
        string = """"Old"
,"cu" """
        with pytest.raises(FailedToCreateNodeFromInput):
            node = GetNodeFromInput(string)


    def testAssignFromSourceNestedTuples(self):
        string = "((1, 2, 2), (2, 2)),  # Different dim, non-zerodim"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceNestedEmptyTuples(self):
        string = "((1, 2), ()),  # one zerodim"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSourceNestedEmptyTuples_2(self):
        string = "((1,2,4),(1)),  # one zerodim"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceNestedEmptyTuples_3(self):
        string = "((1,2),()),  # one zerodim"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSourceNestedEmptyTuples_3_1(self):
        string = "((1,2),()),"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceNestedEmptyTuples_3_2(self):
        string = "((1,2),())"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSourceEmptyTuple(self):
        string = "(),"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSourceEmptyTuple1(self):
        string = "(())"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSourceEmptyTuple2(self):
        string = "((), ()),"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSourceEmptyTuple3(self):
        string = "((()), ())"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    @pytest.mark.skip('issue 324')
    def testAssignFromSourceSubscript(self):
        string = """NSFusionElType = Union[
    Callable,  # call_function or call_module type, example: F.linear or nn.Conv2d
    str,  # call_method name, example: "dequantize"
    Tuple[str, Any],  # call_method name and first argument, example: ("to", torch.float16)
]
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    @pytest.mark.skip('issue 324')
    def testAssignFromSourceSubscript1_1(self):
        string = """Union[
        Callable,  # call_function or call_module type, example: F.linear or nn.Conv2d
        Tuple[str, Any],  # call_method name and first argument, example: ("to", torch.float16)
    ]
    """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    @pytest.mark.skip('issue 324')
    def testAssignFromSourceSubscript1_2(self):
        string = """(
        Callable,  # call_function or call_module type, example: F.linear or nn.Conv2d
        Tuple[str, Any],  # call_method name and first argument, example: ("to", torch.float16)
    )
    """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceSubscript1_3(self):
        string = """(
        Callable,  # call_function 
        Tuple_a,  # call_method 
    )
    """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceSubscript2(self):
        string = """Union[
        Callable, 
        Tuple[str, Any]
    ]
    """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)