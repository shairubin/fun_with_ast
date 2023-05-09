from manipulate_node import create_node
from manipulate_node.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateFunctionDefTest(CreateNodeTestBase):

    def testFunctionDefWithArgs(self):
        expected_string = """def testFunc(a, b):
  pass"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.FunctionDef('testFunc', create_node.arguments(args=['a', 'b']),
                                            body=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testFunctionDefWithStringKwargs(self):
        expected_string = """def testFunc(a='b', c='d'):
  pass"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.FunctionDef(
            'testFunc',
            create_node.arguments(args=['a', 'c'], defaults=[create_node.Str('b'), create_node.Str('d')]),
            body=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testFunctionDefWithArgsAndStringKwargs(self):
        expected_string = """def testFunc(x,y,a='b', c='d'):
  pass"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.FunctionDef(
            'testFunc',
            create_node.arguments(args=['x', 'y', 'a', 'c'], defaults=[create_node.Str('b'), create_node.Str('d')]),
            body=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testFunctionDefWithNameKwargs(self):
        expected_string = """def testFunc(a=b, c=d):
  pass"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.FunctionDef('testFunc', create_node.arguments(args=['a', 'c'], defaults=['b', 'd']),
                                            body=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testFunctionDefWithBody(self):
        expected_string = """def testFunc():
  a"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.FunctionDef(
            'testFunc', body=[create_node.Expr(create_node.Name('a'))])
        self.assertNodesEqual(expected_node, test_node)

    def testFunctionDefWithVararg(self):
        expected_string = """def testFunc(*args):
  pass"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.FunctionDef(
            'testFunc', create_node.arguments(vararg='args'), body=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testFunctionDefWithKwarg(self):
        expected_string = """def testFunc(**kwargs):
  pass"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.FunctionDef(
            'testFunc', create_node.arguments(kwarg='kwargs'), body=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testDecoratorList(self):
        expected_string = """@decorator
@other_decorator(arg)
def testFunc(**kwargs):
  pass"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.FunctionDef(
            'testFunc', create_node.arguments(kwarg='kwargs'), body=[create_node.Pass()],
            decorator_list=[
                create_node.Name('decorator'),
                create_node.Call('other_decorator', ['arg'])
            ])
        self.assertNodesEqual(expected_node, test_node)
