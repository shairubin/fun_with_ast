import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput, FailedToCreateNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ListTest(BaseTestUtils):

    def testCreateNodeFromInput(self):
        string = '[\t   a\t, 1 \t] \t #comment'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput1(self):
        string = '[a, 1,] \t #comment'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput11(self):
        string = '[a, 1]  #comment'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInput2(self):
        string = '[a,]'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput3(self):
        string = '[a,  \t b,   ]'
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
        string = '[-1,]'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputListWithEOL(self):
        string = '[[1,2], [3,4]]'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputListWithEOL2(self):
        string = '[[1,2], [3,4],]'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInputListWithEOL21(self):
        string = '[[1,2], [3,4],]\n'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputListWithEOL22(self):
        string = '[[1,2], [3,4],]\n     '
        node =GetNodeFromInput(string, get_module=True)
        self._assert_match(node, string)

    def testCreateNodeFromInputListWithEOL23(self):
        string = '[1,2]\n     '
        node =GetNodeFromInput(string, get_module=True)
        self._assert_match(node, string)

    def testCreateNodeFromInputListWithEOL24(self):
        string = '[1,2]\n     '
        node =GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string, trim_suffix_spaces=True)

    def testCreateNodeFromInputListWithEOL25(self):
        string = '[1,2]\n     '
        node =GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string, trim_suffix_spaces=False)

    def testCreateNodeFromInputListWithEOL3(self):
        string = '[[1,2],\n [3,4]]'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputListWithEOL31(self):
        string = '[1,\n 2]'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)


    def testCreateNodeFromInputListWithJoinedStr(self):
        string = """[
        *k[:name_idx],
        f"{k[name_idx][:-1]}_{i}",
    ]
"""
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputListWithJoinedStr2(self):
        string = """[
        a,
        f"x",
    ]
"""
        node = GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputListWithMultiLineComments(self):
        string = """[
        a, #comment 1
        f, #comment 2
    ]
"""
        node = GetNodeFromInput(string)
        self._assert_match(node, string)

    def _assert_match(self, node, string):
        self._verify_match(node, string)
    def testAssignFromSourceWithListAsValue(self):
        string = """{"attention_mask": aaa, **mmm}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithListAsValue2(self):
        string = """{"attention_mask": aaa}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithListAsValue3(self):
        string = """{"attention_mask": [a,b]}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithListAsValue31(self):
        string = """{"attention_mask": a,b}"""
        with pytest.raises(FailedToCreateNodeFromInput):
            node = GetNodeFromInput(string)

    def testAssignFromSourceWithListAsValue4(self):
        string = """{"attention_mask": [a, **m]}"""
        with pytest.raises(FailedToCreateNodeFromInput):
            node = GetNodeFromInput(string)

    def testAssignFromSourceWithListAsValue41(self):
        string = """{"attention_mask": a, **m}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAssignFromSourceWithJstr(self):
        string = """[
            f"-DPYTHON_EXECUTABLE:FILEPATH={sys.executable}",
            f"-DPYTHON_INCLUDE_DIR={sysconfig.get_path('include')}",
        ]"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testAssignFromSourceWithJstr2(self):
        string = """[f"-DPYTHON_EXECUTABLE:FILEPATH={sys.executable}",f"-DPYTHON_INCLUDE_DIR={sysconfig.get_path('include')}",]"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testListEllipsis(self):
        string = """[...,1,...]"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testListTrailingComma(self):
        string = "[1,2],"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testListTrailingComma1(self):
        string = "[1,2,],"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testListTrailingComma2(self):
        string = "[(1,2),],"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testListTrailingComma3(self):
        string = "[(1),3,]"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testListTrailingComma4(self):
        string = "[(1,a),3]"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testListTrailingComma4(self):
        string = "[(1,a),(3,b)]"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testListTrailingComma5(self):
        string = "[[1,a],3]"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testListTrailingComma6(self):
        string = "[[1,a],[3,4]],"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
