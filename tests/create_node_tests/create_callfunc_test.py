from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateCallTest(CreateNodeTestBase):

    def testCallWithSimpleCaller(self):
        expected_string = 'a()'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Call('a')
        self.assertNodesEqual(expected_node, test_node)

    def testCallWithDotSeparatedCaller(self):
        expected_string = 'a.b()'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Call('a.b')
        self.assertNodesEqual(expected_node, test_node)

    def testCallWithAttributeNode(self):
        expected_string = 'a.b()'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Call(create_node.VarReference('a', 'b'))
        self.assertNodesEqual(expected_node, test_node)

    def testCallWithAttributeNodeAndParam(self):
        expected_string = 'a.b(\'fun-with-ast\')'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Call('a.b', args=[create_node.Str('fun-with-ast')])
        self.assertNodesEqual(expected_node, test_node)

    def testCallWithArgs(self):
        expected_string = 'a(b)'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Call('a', args=['b'])
        self.assertNodesEqual(expected_node, test_node)

    def testCallWithKwargs(self):
        expected_string = 'a(b="c")'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Call(
            #        'a', keywords=[{'key': 'b','value':'c'}])
            'a', keywords=[create_node.keyword('b', create_node.Str('c'))])
        self.assertNodesEqual(expected_node, test_node)

    def testCallWithStarargsString(self):
        expected_string = 'a(*b)'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Call('a', [create_node.Starred(create_node.Name('b'))])
        self.assertNodesEqual(expected_node, test_node)

    def testCallWithStarargsNode(self):
        expected_string = 'a(*[b])'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Call(
            'a', [create_node.Starred(create_node.List(create_node.Name('b')))])
        self.assertNodesEqual(expected_node, test_node)

    def testCallWithKwargsString(self):
        expected_string = 'a(**b)'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Call(
            'a', keywords=create_node.keyword(None, create_node.Name('b')))
        self.assertNodesEqual(expected_node, test_node)

    def testCallWithKwargsNode(self):
        expected_string = 'a(**{b:c})'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Call(
            'a',
            keywords=create_node.keyword(None, create_node.Dict(('b'), ('c'))))

        #        reate_node.Dict(
        #            keys=(create_node.Name('b'),),
        #            values=(create_node.Name('c'),)
        #        )
        #    )
        self.assertNodesEqual(expected_node, test_node)
