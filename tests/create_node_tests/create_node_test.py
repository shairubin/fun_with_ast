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
from create_node import GetNodeFromInput
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


class CreateAssignTest(CreateNodeTestBase):

    def testAssignWithSimpleString(self):
        expected_string = 'a = "b"'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Assign('a', create_node.Str('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testAssignListWithSimpleString(self):
        expected_string = 'a=c="b"'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Assign(['a', 'c'], create_node.Str('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testAssignWithNode(self):
        expected_string = 'a = "b"'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Assign(
            create_node.Name('a', ctx_type=create_node.CtxEnum.STORE),
            create_node.Str('b'))
        self.assertNodesEqual(expected_node, test_node)

    def testAssignWithTuple(self):
        expected_string = '(a, c) = "b"'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Assign(
            create_node.Tuple(['a', 'c'], ctx_type=create_node.CtxEnum.STORE),
            create_node.Str('b'))
        self.assertNodesEqual(expected_node, test_node)


class CreateDictTest(CreateNodeTestBase):

    def testDictWithStringKeys(self):
        expected_string = '{"a": "b"}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Dict(
            [create_node.Str('a')],
            [create_node.Str('b')])
        self.assertNodesEqual(expected_node, test_node)

    def testDictWithNonStringKeys(self):
        expected_string = '{a: 1}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Dict(
            [create_node.Name('a')],
            [create_node.Num(1)])
        self.assertNodesEqual(expected_node, test_node)

    def testDictWithNoKeysOrVals(self):
        expected_string = '{}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Dict([], [])
        self.assertNodesEqual(expected_node, test_node)

    def testDictRaisesErrorIfNotMatchingLengths(self):
        with self.assertRaises(ValueError):
            unused_test_node = create_node.Dict(
                [create_node.Str('a')],
                [])


class CreateDictComprehensionTest(CreateNodeTestBase):

    def testBasicDictComprehension(self):
        expected_string = '{a: b for c in d}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.DictComp('a', 'b', 'c', 'd')
        self.assertNodesEqual(expected_node, test_node)

    def testBasicDictComprehensionWithIfs(self):
        expected_string = '{a: b for c in d if e < f}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.DictComp(
            'a', 'b', 'c', 'd',
            create_node.Compare('e', '<', 'f'))
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


class CreateIfTest(CreateNodeTestBase):

    def testBasicIf(self):
        expected_string = """if True:\n  pass"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.If(
            create_node.Constant(True),
            body=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testBasicIfElse(self):
        expected_string = """if True:\n  pass\nelse:\n  pass"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.If(conditional=create_node.Constant(True),
                                   body=[create_node.Pass()], orelse=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testBasicIfElif(self):
        expected_string = """if True:
  pass
elif False:
  pass
"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.If(
            create_node.Constant(True),
            body=[create_node.Pass()],
            orelse=[create_node.If(create_node.Constant(False), body=[create_node.Pass()])])
        self.assertNodesEqual(expected_node, test_node)

    def testIfInElse(self):
        expected_string = """if True:
  pass
else:
  if False:
    pass
"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.If(
            create_node.Constant(True), body=[create_node.Pass()],
            orelse=[create_node.If(conditional=create_node.Constant(False), body=[create_node.Pass()])])
        self.assertNodesEqual(expected_node, test_node)

    def testIfAndOthersInElse(self):
        expected_string = """if True:
  pass
else:
  if False:
    pass
  True
"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.If(
            create_node.Constant(True), body=[create_node.Pass()],
            orelse=[create_node.If(conditional=create_node.Constant(False), body=[create_node.Pass()]),
                    create_node.Expr(create_node.Constant(True))])
        self.assertNodesEqual(expected_node, test_node)


class CreateIfExpTest(CreateNodeTestBase):

    def testBasicIfExp(self):
        expected_string = """a if True else b"""
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.IfExp(
            create_node.Constant(True), create_node.Name('a'), create_node.Name('b'))
        self.assertNodesEqual(expected_node, test_node)


class CreateImportTest(CreateNodeTestBase):

    def testImport(self):
        expected_string = """import foo"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Import(import_part='foo')
        self.assertNodesEqual(expected_node, test_node)

    def testImportAs(self):
        expected_string = """import foo as foobar"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Import(import_part='foo', asname='foobar')
        self.assertNodesEqual(expected_node, test_node)

    def testImportFrom(self):
        expected_string = """from bar import foo"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Import(import_part='foo', from_part='bar')
        self.assertNodesEqual(expected_node, test_node)

    def testImportFromAs(self):
        expected_string = """from bar import foo as foobar"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Import(
            import_part='foo', from_part='bar', asname='foobar')
        self.assertNodesEqual(expected_node, test_node)


class CreateListComprehensionTest(CreateNodeTestBase):

    def testBasicListComprehension(self):
        expected_string = '[a for a in b]'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.ListComp('a', 'a', 'b')
        self.assertNodesEqual(expected_node, test_node)

    def testBasicListComprehensionWithIfs(self):
        expected_string = '[a for a in b if c < d]'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.ListComp(
            'a', 'a', 'b',
            create_node.Compare('c', '<', 'd'))
        self.assertNodesEqual(expected_node, test_node)


class CreateNumTest(CreateNodeTestBase):

    def testNumWithInteger(self):
        expected_string = '15'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Num(15)
        self.assertNodesEqual(expected_node, test_node)

    def testNumWithHex(self):
        expected_string = '0xa5'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Num(0xa5)
        self.assertNodesEqual(expected_node, test_node)

    def testNumWithFloat(self):
        expected_string = '0.25'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Constant(0.25)
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


class CreateSetComprehensionTest(CreateNodeTestBase):

    def testBasicSetComprehension(self):
        expected_string = '{a for a in b}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.SetComp('a', 'a', 'b')
        self.assertNodesEqual(expected_node, test_node)

    def testBasicSetComprehensionWithIfs(self):
        expected_string = '{a for a in b if c < d}'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.SetComp(
            'a', 'a', 'b',
            create_node.Compare('c', '<', 'd'))
        self.assertNodesEqual(expected_node, test_node)


class CreateReturnTest(CreateNodeTestBase):

    def testRetrunSigleValueInt(self):
        expected_string = 'return 1'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Return(1)
        self.assertNodesEqual(expected_node, test_node)

    def testRetrunSigleValueStr(self):
        expected_string = "return '1'"
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Return('1')
        self.assertNodesEqual(expected_node, test_node)

    def testRetrunSigleValueName(self):
        expected_string = "return a"
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Return(create_node.Name('a'))
        self.assertNodesEqual(expected_node, test_node)

class CreateStrTest(CreateNodeTestBase):

    def testStr(self):
        expected_string = '"a"'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Str('a')
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
