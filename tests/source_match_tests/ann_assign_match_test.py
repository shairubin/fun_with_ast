from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from tests.source_match_tests.base_test_utils import BaseTestUtils


class AnnAssignMatcherTest(BaseTestUtils):

    def testAnnAssignFromSource(self):
        string = 'a: int'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAnnAssignFromSource2(self):
        string = 'a: str'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnAssignFromSource3(self):
        string = 'a: MyType'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnAssignFromSourceWithValue(self):
        string = 'a: int = 1'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAnnAssignFromSourceWithValue2(self):
        string = "a: str = param"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAnnAssignFromSourceWithValue3(self):
        string = "a: str ='string'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAnnAssignFromSourceWithValue4(self):
        string = "a: str =     'string'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnAssignFromSourceWithValue5(self):
        string = "a: str   = \t   'string' # comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAnnAssignFromSourceWithValue6(self):
        string = "a: str   = \t   \"string\" # comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnAssignFromSourceWithComment(self):
        string = 'a: int # this is a comment'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnAssignFromSourceWithModule(self):
        string = 'self.exec_info: Optional[_ExecutionInfo] = None\n  \ndef foo():\n  pass'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testAnnAssignFromSourceWithModule2(self):
        string = 'a: int =1\n   \nb: int =2  \n  '
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testAnnAssignFromSourceWithModule3(self):
        string = """class bar():
        def foo1(): 
          a: int =1   
        def foo2(): 
          b: int =2
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def _assert_matched_source(self, node, string):
        self._verify_match(node, string)
