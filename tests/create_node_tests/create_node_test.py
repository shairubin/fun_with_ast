"""Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Tests for create_node.py
"""

import _ast
import unittest

import create_node
from fun_with_ast.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase




###############################################################################
# Node Creators
###############################################################################


class CreateAssertTest(CreateNodeTestBase):

    def testBasicAssert(self):
        expected_string = 'assert a'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Assert(
            create_node.Name('a'))
        self.assertNodesEqual(expected_node, test_node)

    def testAssertWithMessage(self):
        expected_string = 'assert a, "a failure"'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Assert(
            create_node.Name('a'),
            create_node.Str('a failure'))
        self.assertNodesEqual(expected_node, test_node)


class CreateGeneratorExpTest(CreateNodeTestBase):

    def testBasicSetComprehension(self):
        expected_string = '(a for a in b)'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.GeneratorExp('a', 'a', 'b')
        self.assertNodesEqual(expected_node, test_node)

    def testBasicGeneratorExpWithIfs(self):
        expected_string = '(a for a in b if c < d)'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.GeneratorExp(
            'a', 'a', 'b',
            create_node.Compare('c', '<', 'd'))
        self.assertNodesEqual(expected_node, test_node)


class CreateIfExpTest(CreateNodeTestBase):

    def testBasicIfExp(self):
        expected_string = """a if True else b"""
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.IfExp(
            create_node.Constant(True), create_node.Name('a'), create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)






class CreatePassTest(CreateNodeTestBase):

    def testPass(self):
        expected_string = 'pass'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Pass()
        self.assertNodesEqual(expected_node, test_node)


class CreateSetTest(CreateNodeTestBase):

    def testSet(self):
        expected_string = '{"a", "b"}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Set(
            create_node.Str('a'),
            create_node.Str('b'))
        self.assertNodesEqual(expected_node, test_node)






class CreateExceptionHandlerTest(CreateNodeTestBase):

    def testExceptionHadlerBasic(self):
        string = """try:
 a
except:
 pass
"""
        expected_node = GetNodeFromInput(string).handlers[0]
        test_node = create_node.ExceptHandler(body=[(create_node.Pass())])
        self.assertNodesEqual(expected_node, test_node)


class CreateTryTest(CreateNodeTestBase):

    def testTryBasic(self):
        string = """try:
  pass
except:
  pass
"""
        expected_node = GetNodeFromInput(string)
        test_node = create_node.Try(body=[(create_node.Pass())],
                                    except_handlers=[create_node.ExceptHandler(None, None, [create_node.Pass()])])
        self.assertNodesEqual(expected_node, test_node)

    def testTrySpecificExcept(self):
        string = """try:
  pass
except ValueError:
  pass
"""
        expected_node = GetNodeFromInput(string)
        test_node = create_node.Try(body=[(create_node.Pass())], except_handlers=[
            create_node.ExceptHandler(create_node.Name('ValueError'), None, [create_node.Pass()])])
        self.assertNodesEqual(expected_node, test_node)


###############################################################################
# Tests for Multiple-Node Creators
###############################################################################


class GetCtxTest(CreateNodeTestBase):

    def testGetLoad(self):
        self.assertIsInstance(create_node.GetCtx(create_node.CtxEnum.LOAD),
                              _ast.Load)

    def testGetStore(self):
        self.assertIsInstance(create_node.GetCtx(create_node.CtxEnum.STORE),
                              _ast.Store)

    def testGetDel(self):
        self.assertIsInstance(create_node.GetCtx(create_node.CtxEnum.DEL),
                              _ast.Del)


#  def testGetParam(self):
#    self.assertIsInstance(create_node.GetCtx(create_node.CtxEnum.PARAM),
#                          _ast.Param)


if __name__ == '__main__':
    unittest.main()
