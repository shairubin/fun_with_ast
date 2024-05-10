from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class RetrunMatcherTest(BaseTestUtils):
    def testSimpleReturn(self):
        node = create_node.Return(1)
        string = 'return 1'
        self._assert_match(node, string)

    def testSimpleReturnFromString(self):
        string = 'return 1'
        node = GetNodeFromInput(string)
        self._assert_match(node, string)
    def testSimpleReturnFromStringWithNL(self):
        string = 'return 1\n'
        node = GetNodeFromInput(string)
        self._assert_match(node, string)

    def testSimpleReturnFromStringWithComment(self):
        string = 'return 1 #comment'
        node = GetNodeFromInput(string)
        self._assert_match(node, string)
    def testSimpleReturnFromStringWithCommentNL(self):
        string = 'return 1 #comment\n'
        node = GetNodeFromInput(string)
        self._assert_match(node, string)


    def testSimpleReturnFromString2(self):
        string = 'return  isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)'
        node = GetNodeFromInput(string)
        self._assert_match(node, string)


    def testReturnStr(self):
        node = create_node.Return("1", "'")
        string = "return '1'"
        self._assert_match(node, string)

    def testReturnStrDoubleQuote(self):
        node = create_node.Return('1', "\"")
        string = "return \"1\""
        self._assert_match(node, string)

    def testReturnName(self):
        node = create_node.Return(create_node.Name('a'))
        string = "return a"
        self._assert_match(node, string)

    def testReturnTuple(self):
        node = create_node.Return(create_node.Tuple(['a', 'b']))
        string = "return (a,b)"
        self._assert_match(node, string)

    def testReturnTupleNoParans(self):
        node = create_node.Return(create_node.Tuple(['a', 'b']))
        string = "return a,b"
        self._assert_match(node, string)


    #pytest.mark.skip(reason="issue 303")
    def testStringWithParathesis(self):
        string = """def foo(): 
        return (
        a.b  # type:ignore[attr-defined
    )"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    #@pytest.mark.skip(reason="issue 303")
    def testStringWithParathesis_2(self):
        string = """return (a.b  # type:ignore[attr-defined]
        )"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    #@pytest.mark.skip(reason="issue 303")
    def testStringWithParathesis_2_1(self):
        string = """return (0 # comment
        )"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testStringWithParathesis_3(self):
        string = """return a.b  # type:ignore[attr-defined]
        """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testStringWithParathesis_4(self):
        string = """return 0  # type:ignore[attr-defined]
        """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def _assert_match(self, node, string):
        self._verify_match(node, string)