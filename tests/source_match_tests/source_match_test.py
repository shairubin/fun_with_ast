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


Tests for py
"""

import unittest

import pytest
from fun_with_ast.placeholders.list_placeholder import ListFieldPlaceholder
from fun_with_ast.placeholders.list_placeholder import SeparatedListFieldPlaceholder
from fun_with_ast.placeholders.composite import FieldPlaceholder
from fun_with_ast.placeholders.text import TextPlaceholder
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
import fun_with_ast.manipulate_node.create_node as create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.source_matchers import defualt_matcher
from tests.source_match_tests.base_test_utils import BaseTestUtils

DEFAULT_TEXT = 'default'


class TextPlaceholderTest(unittest.TestCase):

    def testMatchSimpleText(self):
        placeholder = TextPlaceholder('.*', DEFAULT_TEXT)
        matched_text = placeholder._match(None, 'to match')
        self.assertEqual(matched_text, 'to match')
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, 'to match')

    def testPartialMatchEnd(self):
        placeholder = TextPlaceholder(r'def \(', DEFAULT_TEXT)
        matched_text = placeholder._match(None, 'def (foo')
        self.assertEqual(matched_text, 'def (')
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, 'def (')

    def testMatchWithoutMatchingReturnsDefault(self):
        placeholder = TextPlaceholder('.*', DEFAULT_TEXT)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, DEFAULT_TEXT)

    def testCantMatchThrowsError(self):
        placeholder = TextPlaceholder('doesnt match', DEFAULT_TEXT)
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder._match(None, 'to match')

    def testMatchWhitespace(self):
        whitespace_text = '  \t \n  '
        placeholder = TextPlaceholder(r'\s*')
        matched_text = placeholder._match(None, whitespace_text)
        self.assertEqual(matched_text, whitespace_text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, whitespace_text)

    def testWhitespaceMatchesLineContinuations(self):
        whitespace_text = '  \t \n \\\n  \\\n  '
        placeholder = TextPlaceholder(r'\s*')
        matched_text = placeholder._match(None, whitespace_text)
        self.assertEqual(matched_text, whitespace_text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, whitespace_text)

    def testWhitespaceMatchesComments(self):
        whitespace_text = '  \t # abc\n  '
        placeholder = TextPlaceholder(r'\s*')
        matched_text = placeholder._match(None, whitespace_text)
        self.assertEqual(matched_text, whitespace_text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, whitespace_text)

    def testMultipleStatementsSeparatedBySemicolon(self):
        whitespace_text = 'pdb;pdb'
        placeholder = TextPlaceholder(r'pdb\npdb')
        matched_text = placeholder._match(None, whitespace_text)
        self.assertEqual(matched_text, whitespace_text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, whitespace_text)

    def testCommentAfterExpectedLinebreak(self):
        whitespace_text = 'pdb  #  \t A comment\n'
        placeholder = TextPlaceholder(r'pdb\n')
        matched_text = placeholder._match(None, whitespace_text)
        self.assertEqual(matched_text, whitespace_text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, whitespace_text)

    def testCommentInNewLine(self):
        text = '\n   #  A comment\n'
        placeholder = TextPlaceholder('\n   #  A comment\n')
        matched_text = placeholder._match(None, text)
        self.assertEqual(matched_text, text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, text)


class FieldPlaceholderTest(unittest.TestCase):

    def testMatchSimpleFieldWithSpace(self):
        node = create_node.Name('foobar')
        placeholder = FieldPlaceholder('id')
        matched_text = placeholder._match(node, 'foobar\t')
        self.assertEqual(matched_text, 'foobar')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar')
        matched_text = placeholder._match(node, 'foobar\t\t\n')
        self.assertEqual(matched_text, 'foobar')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar')
        with self.assertRaises(BadlySpecifiedTemplateError):
            matched_text = placeholder._match(node, ' foobar\t\t\n')
            self.assertEqual(matched_text, 'foobar')
            test_output = placeholder.GetSource(node)
            self.assertEqual(test_output, 'foobar')

    def testMatchSimpleField(self):
        node = create_node.Name('foobar')
        placeholder = FieldPlaceholder('id')
        matched_text = placeholder._match(node, 'foobar')
        self.assertEqual(matched_text, 'foobar')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar')

    def testPartialMatch(self):
        node = create_node.Name('bar')
        placeholder = FieldPlaceholder(
            'id', before_placeholder=TextPlaceholder('foo'))
        matched_text = placeholder._match(node, 'foobarbaz')
        self.assertEqual(matched_text, 'foobar')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar')

    def testBeforePlaceholder(self):
        node = create_node.Name('bar')
        placeholder = FieldPlaceholder(
            'id',
            before_placeholder=TextPlaceholder('before '))
        matched_text = placeholder._match(node, 'before bar')
        self.assertEqual(matched_text, 'before bar')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'before bar')

    def testCantMatchThrowsError(self):
        node = create_node.Name('doesnt_match')
        placeholder = FieldPlaceholder('id')
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder._match(node, 'to match')

    def testRaisesErrorIfFieldIsList(self):
        node = create_node.FunctionDef('function_name')
        placeholder = FieldPlaceholder('body')
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder._match(node, 'invalid_match')

    def testChangingValueChangesOutput(self):
        node = create_node.Name('bar')
        placeholder = FieldPlaceholder(
            'id', before_placeholder=TextPlaceholder('foo'))
        matched_text = placeholder._match(node, 'foobarbaz')
        self.assertEqual(matched_text, 'foobar')
        node.id = 'hello'
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foohello')

    def testWithoutMatch(self):
        node = create_node.Name('bar')
        placeholder = FieldPlaceholder('id')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'bar')


class ListFieldPlaceholderTest(unittest.TestCase):

    def testMatchSimpleField(self):
        body_node = create_node.Expr(create_node.Name('foobar'))
        node = create_node.FunctionDef('function_name', body=[body_node])
        placeholder = ListFieldPlaceholder('body')
        matched_text = placeholder._match(node, 'foobar\n')
        self.assertEqual(matched_text, 'foobar\n')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar\n')

    def testMultipleListItems(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        placeholder = ListFieldPlaceholder('body')
        matched_text = placeholder._match(node, 'foobar\nbaz\n')
        self.assertEqual(matched_text, 'foobar\nbaz\n')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar\nbaz\n')

    def testMultipleListItemsBeginningAndEnd(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        placeholder = ListFieldPlaceholder(
            'body',
            before_placeholder=TextPlaceholder('z'),
            after_placeholder=TextPlaceholder('zz'))
        matched_text = placeholder._match(node, 'zfoobar\nzzzbaz\nzz')
        self.assertEqual(matched_text, 'zfoobar\nzzzbaz\nzz')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'zfoobar\nzzzbaz\nzz')

    def testMatchRaisesErrorIfFieldIsNotList(self):
        node = create_node.Name('bar')
        placeholder = ListFieldPlaceholder(
            'id', before_placeholder=TextPlaceholder('\n', '\n'),
            exclude_first_before=True)
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder._match(node, 'foobar\nbaz')

    def testMatchRaisesErrorIfFieldDoesntMatch(self):
        body_node = create_node.Expr(create_node.Name('foobar'))
        node = create_node.FunctionDef('function_name', body=[body_node])
        placeholder = ListFieldPlaceholder(
            'body', before_placeholder=TextPlaceholder('\n', '\n'),
            exclude_first_before=True)
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder._match(node, 'no match here')

    def testMatchRaisesErrorIfSeparatorDoesntMatch(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        placeholder = ListFieldPlaceholder(
            'body', before_placeholder=TextPlaceholder('\n', '\n'),
            exclude_first_before=True)
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder._match(node, 'foobarbaz')

    # TODO: Renabled this after adding indent information to matchers
    @unittest.expectedFailure
    def testListDefaults(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        module_node = create_node.Module(node)
        placeholder = ListFieldPlaceholder(
            'body', before_placeholder=TextPlaceholder('', ', '),
            exclude_first_before=True)
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, '  foobar\n,   baz\n')

class SeparatedListFieldPlaceholderTest(unittest.TestCase):

    def testMatchSepertedListSingleElement(self):
        node = create_node.Assign('foo', 1)
        placeholder = SeparatedListFieldPlaceholder('targets',
                                                                 after__separator_placeholder=TextPlaceholder(r'[ \t]*=[ \t]*', '='))
        matched_text = placeholder._match(node, 'foo=1')
        self.assertEqual(matched_text, 'foo=')
        placeholder = FieldPlaceholder('value')
        matched_text = placeholder._match(node, '1')
        self.assertEqual(matched_text, '1')

    def testMatchSepertedListSingleElementWithWS(self):
        node = create_node.Assign('foo', 1)
        placeholder = SeparatedListFieldPlaceholder('targets',
                                                                 after__separator_placeholder=TextPlaceholder(r'[ \t]*=[ \t]*', '='))
        matched_text = placeholder._match(node, 'foo \t   =\t  1')
        self.assertEqual(matched_text, 'foo \t   =\t  ')
        placeholder = FieldPlaceholder('value')
        matched_text = placeholder._match(node, '1')
        self.assertEqual(matched_text, '1')

    def testMatchSepertedListSingleElementWithWSWithComment(self):
        node = create_node.Assign('foo', 1)
        placeholder = SeparatedListFieldPlaceholder('targets',
                                                                 after__separator_placeholder=TextPlaceholder(r'[ \t]*=[ \t]*', '='))
        matched_text = placeholder._match(node, 'foo \t   =\t  1 # comment')
        self.assertEqual(matched_text, 'foo \t   =\t  ')
        placeholder = FieldPlaceholder('value')
        matched_text = placeholder._match(node, '1')
        self.assertEqual(matched_text, '1')

    def testMatchSepertedList(self):
        node = create_node.Assign(['foo', 'bar'], 2)
        placeholder = SeparatedListFieldPlaceholder('targets',
                                                                 after__separator_placeholder=TextPlaceholder(r'[ \t]*=[ \t]*', '='))
        matched_text = placeholder._match(node, 'foo=bar=2')
        self.assertEqual(matched_text, 'foo=bar=')
        placeholder = FieldPlaceholder('value')
        matched_text = placeholder._match(node, '2')
        self.assertEqual(matched_text, '2')


class TestDefaultSourceMatcher(BaseTestUtils):

    def testInvalidExpectedPartsType(self):
        node = create_node.Name('bar')
        with self.assertRaises(ValueError):
            defualt_matcher.DefaultSourceMatcher(node, ['blah'])

    def testBasicTextMatch(self):
        matcher = defualt_matcher.DefaultSourceMatcher(
            None, [TextPlaceholder('blah', DEFAULT_TEXT)])
        matcher.do_match('blah')
        self.assertEqual(matcher.GetSource(), 'blah')

    def testRaisesErrorIfNoTextMatch(self):
        matcher = defualt_matcher.DefaultSourceMatcher(
            None, [TextPlaceholder('blah', DEFAULT_TEXT)])
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.do_match('bla')

    def testBasicFieldMatch(self):
        node = create_node.Name('bar')
        matcher = defualt_matcher.DefaultSourceMatcher(
            node, [FieldPlaceholder('id')])
        matcher.do_match('bar')
        self.assertEqual(matcher.GetSource(), 'bar')

    def testRaisesErrorIfNoFieldMatch(self):
        node = create_node.Name('bar')
        matcher = defualt_matcher.DefaultSourceMatcher(
            node, [FieldPlaceholder('id')])
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.do_match('ba')

    def testBasicListMatch(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        matcher = defualt_matcher.DefaultSourceMatcher(
            node, [ListFieldPlaceholder('body')])
        matcher.do_match('foobar\nbaz\n')
        self.assertEqual(matcher.GetSource(), 'foobar\nbaz\n')

    def testRaisesErrorWhenNoMatchInBasicList(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        matcher = defualt_matcher.DefaultSourceMatcher(
            node, [ListFieldPlaceholder('body')])
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.do_match('foobar\nba\n')

    def testBasicListMatchWhenChangedFieldValue(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        matcher = defualt_matcher.DefaultSourceMatcher(
            node,
            [ListFieldPlaceholder('body')])
        matcher.do_match('foobar\nbaz\n')
        self.assertEqual(matcher.GetSource(), 'foobar\nbaz\n')

    def testAdvancedMatch(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        matcher = defualt_matcher.DefaultSourceMatcher(
            node,
            [TextPlaceholder('def ', 'def '),
             FieldPlaceholder('name'),
             TextPlaceholder(r'\(\)', r'()'),
             ListFieldPlaceholder('body')])
        matcher.do_match('def function_name()foobar\nbaz\n')
        self.assertEqual(matcher.GetSource(), 'def function_name()foobar\nbaz\n')

    # TODO: Renabled this after adding indent information to matchers
    @unittest.expectedFailure
    def testGetSourceWithoutMatchUsesDefaults(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        module_node = create_node.Module(node)
        matcher = defualt_matcher.DefaultSourceMatcher(
            node,
            [TextPlaceholder('def ', 'default '),
             FieldPlaceholder('name'),
             TextPlaceholder(r'\(\)', r'()'),
             SeparatedListFieldPlaceholder(
                 'body', TextPlaceholder('\n', ', '))])
        node.body[0].value.id = 'hello'
        self.assertEqual(matcher.GetSource(),
                         'default function_name()  hello\n,   baz\n')


class TestGetMatcher(unittest.TestCase):

    def testDefaultMatcher(self):
        node = create_node.VarReference('foo', 'bar')
        matcher = GetDynamicMatcher(node)
        matcher.do_match('foo.bar')
        self.assertEqual(matcher.GetSource(), 'foo.bar')


class AttributeMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.VarReference('a', 'b')
        string = 'a.b'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testTripleReferenceMatch(self):
        node = create_node.VarReference('a', 'b', 'c')
        string = 'a.b.c'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())


class ClassDefMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.ClassDef('TestClass', body=[create_node.Pass()])
        string = 'class TestClass():\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchBases(self):
        node = create_node.ClassDef(
            'TestClass', bases=['Base1', 'Base2'], body=[create_node.Pass()])
        string = 'class TestClass(Base1, Base2):\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchBody(self):
        node = create_node.ClassDef(
            'TestClass', body=[create_node.Expr(create_node.Name('a'))])
        string = 'class TestClass():\n  a\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchDecoratorList(self):
        node = create_node.ClassDef(
            'TestClass',
            decorator_list=[create_node.Name('dec'),
                            create_node.Call('dec2')], body=[create_node.Pass()])
        string = '@dec\n@dec2()\nclass TestClass():\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testComplete(self):
        node = create_node.ClassDef(
            'TestClass',
            bases=['Base1', 'Base2'],
            body=[create_node.Expr(create_node.Name('a'))],
            decorator_list=[create_node.Name('dec'),
                            create_node.Call('dec2')])
        string = '@dec\n@dec2()\nclass TestClass(Base1, Base2):\n  a\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())




class CompareMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Lt(),
            create_node.Name('b'))
        string = 'a < b'
        self._assert_match(node, string)

    def testMultiMatch(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Lt(),
            create_node.Name('b'),
            create_node.Lt(),
            create_node.Name('c'))
        string = 'a < b < c'
        self._assert_match(node, string)

    def testEq(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Eq(),
            create_node.Name('b'))
        string = 'a == b'
        self._assert_match(node, string)

    def testNotEq(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.NotEq(),
            create_node.Name('b'))
        string = 'a != b'
        self._assert_match(node, string)

    def testLt(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Lt(),
            create_node.Name('b'))
        string = 'a < b'
        self._assert_match(node, string)

    def testLtE(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.LtE(),
            create_node.Name('b'))
        string = 'a <= b'
        self._assert_match(node, string)

    def testGt(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Gt(),
            create_node.Name('b'))
        string = 'a > b'
        self._assert_match(node, string)

    def testGtE(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.GtE(),
            create_node.Name('b'))
        string = 'a >= b'
        self._assert_match(node, string)

    def testIs(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Is(),
            create_node.Name('b'))
        string = 'a is b'
        self._assert_match(node, string)

    def testIsNot(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.IsNot(),
            create_node.Name('b'))
        string = 'a is not b'
        self._assert_match(node, string)


    def testIn(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.In(),
            create_node.Name('b'))
        string = 'a in b'
        self._assert_match(node, string)

    def testNotIn(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.NotIn(),
            create_node.Name('b'))
        string = 'a not in b'
        self._assert_match(node, string)

    def _assert_match(self, node, string):
        #matcher = GetDynamicMatcher(node)
        #matcher.do_match(string)
        #self.assertEqual(string, matcher.GetSource())
        self._verify_match(node, string)

class ComprehensionMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.comprehension('a', 'b', False)
        string = 'for a in b'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithIf(self):
        node = create_node.comprehension(
            'a', 'b', True,
            create_node.Compare('c', '<', 'd'))
        string = 'for a in b if c < d'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())


class DictMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Dict([create_node.Name('a')],
                                [create_node.Name('b')])
        string = '{a: b}'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testEmptyMatch(self):
        node = create_node.Dict()
        string = '{}'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testTwoItemMatch(self):
        node = create_node.Dict(
            [create_node.Name('a'), create_node.Constant('c',"\"")],
            [create_node.Name('b'), create_node.Constant('d', "\"")])
        string = '{a: b, "c": "d"}'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testTwoItemNoMatch(self):
        node = create_node.Dict(
            [create_node.Name('a'), create_node.Constant('c',"'")],
            [create_node.Name('b'), create_node.Constant('d', "\"")])
        string = '{a: b, "c": "d"}'
    def testTwoItemNoMatch2(self):
        node = create_node.Dict(
            [create_node.Name('a'), create_node.Constant('c',"\"")], # TODO: no need for the extra parameter
            [create_node.Name('b'), create_node.Constant('d', "'")])
        string = "{a: b, 'c': \"d\"}"
        self._verify_match(node,string)
    def testTwoItemNoMatch3(self):
        node = create_node.Dict(
            [create_node.Name('a'), create_node.Constant('c',"\"")],
            [create_node.Name('b'), create_node.Constant('d', "'")])
        string = "{a: b, 'c': 'd'}"
        self._verify_match(node,string)
    def testTwoItemNoMatch3(self):
        node = create_node.Dict(
            [create_node.Name('a'), create_node.Constant('c',"\"")],
            [create_node.Name('b'), create_node.Constant('d', "'")])
        string = "{a: b, 'c\": 'd'}"
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node,string)


class DictComprehensionMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.DictComp('e', 'f', 'a', 'b')
        string = '{e: f for a in b}'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithIf(self):
        node = create_node.DictComp(
            'e', 'f', 'a', 'b',
            create_node.Compare('c', '<', 'd'))
        string = '{e: f for a in b if c < d}'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())


class ExceptHandlerMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.ExceptHandler()
        string = 'except:\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithType(self):
        node = create_node.ExceptHandler('TestException')
        string = 'except TestException:\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithName(self):
        node = create_node.ExceptHandler('TestException', name='as_part')
        string = 'except TestException as as_part:\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithBody(self):
        node = create_node.ExceptHandler(
            body=[create_node.Expr(create_node.Name('a'))])
        string = 'except:\n  a\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())


class IfExpMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.IfExp(
            create_node.Name('True'), create_node.Name('a'), create_node.Name('b'))
        string = 'a if True else b'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    @pytest.mark.skip(reason="Not Implemented Yet")
    def testChangeParts(self):
        node = create_node.IfExp(
            create_node.Name('True'), create_node.Name('a'), create_node.Name('b'))
        string = 'a if True else b'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        node.test = create_node.Name('False')
        node.body = create_node.Name('c')
        node.orelse = create_node.Name('d')
        self.assertEqual('c if False else d', matcher.GetSource())


class ListComprehensionMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.ListComp('c', 'a', 'b')
        string = '[c for a in b]'
        self._assert_match(node, string)

    def _assert_match(self, node, string):
        #matcher = GetDynamicMatcher(node)
        #matcher.do_match(string)
        #self.assertEqual(string, matcher.GetSource())
        self._verify_match(node, string)
    def testBasicMatchWithIf(self):
        node = create_node.ListComp(
            'c', 'a', 'b',
            create_node.Compare('c', '<', 'd'))
        string = '[c for a in b if c < d]'
        self._assert_match(node, string)


class SetComprehensionMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.SetComp('c', 'a', 'b')
        string = '{c for a in b}'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithIf(self):
        node = create_node.SetComp(
            'c', 'a', 'b',
            create_node.Compare('c', '<', 'd'))
        string = '{c for a in b if c < d}'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())


class StrMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Constant('foobar', "\"")
        string = '"foobar"'
        self._match_string(node, string)

    def _match_string(self, node, string):
        self._verify_match(node, string)

    def testPrefixMatch(self):
        node = create_node.Constant('foobar', "\"")
        string = 'r"foobar"'
        self._match_string(node, string)

    def testQuoteWrapped(self):
        node = create_node.Constant('foobar', "\"")
        string = '("foobar")'
        self._match_string(node, string)

    def testContinuationMatch(self):
        node = create_node.Constant('foobar', "\"")
        string = '"foo"\n"bar"'
        self._match_string(node, string)


    def testContinuationMatch3(self):
        node = create_node.Constant('foobar', "'")
        string = "'foo''bar'"
        self._match_string(node, string)

    def testContinuationMatch2(self):
        node = create_node.Constant('foobar', "'")
        string = "'fo''o''b''a''r'"
        self._match_string(node, string)

    def testBasicTripleQuoteMatch(self):
        node = create_node.Str('foobar')
        string = '"""foobar"""'

    def testMultilineTripleQuoteMatch(self):
        node = create_node.Constant('foobar\n\nbaz', "\"\"\"")
        string = '"""foobar\n\nbaz"""'
        self._match_string(node, string)

    def testQuoteTypeMismatch(self):
        node = create_node.Constant('foobar', "\"")
        string = '"foobar\''
        matcher = GetDynamicMatcher(node)
        with self.assertRaises(ValueError):
            matcher.do_match(string)

    def testSChange(self):
        node = create_node.Constant('foobar', "\"")
        string = '"foobar"'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        node.s = 'hello'
        self.assertEqual('"hello"', matcher.GetSource())

    def testSChangeInContinuation(self):
        node = create_node.Constant('foo\nbar', "'")
        string = "'foo\nbar'"
        self._match_string(node, string)

    def testQuoteTypeChangeToTripleQuote(self):
        node = create_node.Constant('foobar', "\"")
        string = '\"\"\"foobar\"\"\"'
        self._match_string(node, string)


class CommentMatcherTest(unittest.TestCase):
    def testBasicMatch(self):
        node = create_node.Comment('#comment')
        string = '#comment'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchSpecialChars(self):
        string = '##\t#   c \to m \t  m e n t ? # $'
        node = create_node.Comment(string)
        matcher = GetDynamicMatcher(node)
        matched_text = matcher.do_match(string)
        self.assertEqual(string, matched_text)


class TryFinallyMatcherTest(unittest.TestCase):
    # no exception handlers - not valid in python 3 ?
    #   def testBasicMatch(self):
    #     node = create_node.Try(
    #         [create_node.Expr(create_node.Name('a'))],
    #         [create_node.Expr(create_node.Name('c'))])
    #     string = """try:
    #   a
    # finally:
    #   c
    # """
    #    matcher = GetDynamicMatcher(node)
    #    matcher.do_match(string)
    #    self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithExcept(self):
        node = create_node.Try(
            [create_node.Expr(create_node.Name('a'))],
            [create_node.ExceptHandler()],
            [create_node.Expr(create_node.Name('c'))])
        string = """try:
  a
except:
  pass
finally:
  
  
  c
"""
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithExceptAndAs(self):
        node = create_node.Try(
            [create_node.Expr(create_node.Name('a'))],
            [create_node.ExceptHandler('Exception2', 'e')],
            [create_node.Expr(create_node.Name('c'))])
        string = """try:
      a 
    except Exception2 as e:
      pass
      
    finally:


      c 
"""
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())


class WithItemMatcherTest(unittest.TestCase):

    def testBasicWithItem(self):
        node = create_node.withitem('a')
        string = 'a'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testWithItemWithAs(self):
        node = create_node.withitem('a', optional_vars='b')
        string = 'a    as     b'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)


class WithMatcherTest(unittest.TestCase):
    # start here next time
    def testBasicWith(self):
        node = create_node.With(
            [create_node.withitem('a')], [create_node.Pass()])
        string = 'with a:\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicWithAs(self):
        node = create_node.With([create_node.withitem('a', optional_vars='b')], [create_node.Pass()])
        string = 'with a as b:\n  pass\n'
        self._assert_match(node, string)

    def testWithAsTuple(self):
        node = create_node.With([create_node.withitem('a', optional_vars=create_node.Tuple(['b', 'c']))],
                                [create_node.Pass()])
        string = 'with   a  as     (b,  c):  \n  pass\n'
        self._assert_match(node, string)

    def _assert_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    # not relevant when using withitem
    # def testChangeWithAsTuple(self):
    #     node = create_node.With([create_node.withitem('a', optional_vars=create_node.Tuple(['b', 'c']))],
    #                             [create_node.Pass()])
    #     string = 'with a as (b, c):\n  pass\n'
    #     matcher = GetDynamicMatcher(node)
    #     matcher.do_match(string)
    #     node.context_expr = create_node.Name('d')
    #     node.optional_vars.elts[0] = create_node.Name('e')
    #     node.optional_vars.elts[1] = create_node.Name('f')
    #     self.assertEqual('with d as (e, f):\n  pass\n', matcher.GetSource())

    def testCompoundWith(self):
        node = create_node.With(
            [create_node.withitem('a', optional_vars='c'), create_node.withitem('b', optional_vars='d')],
            [create_node.Pass()])
        # node = create_node.With(
        #     create_node.Name('a'),
        #     as_part=create_node.Name('c'),
        #     body=[
        #         create_node.With(
        #             create_node.Name('b'),
        #             as_part=create_node.Name('d')
        #         )]
        # )
        string = """with  a as c,  b as d:
  pass
"""
        self._assert_match(node, string)

    # TODO: Renabled this after adding indent information to matchers
    # @unittest.expectedFailure
    # def testCompoundWithReplacements(self):
    #     node = create_node.With(
    #         create_node.Name('a'),
    #         as_part=create_node.Name('c'),
    #         body=[
    #             create_node.With(
    #                 create_node.Name('b'),
    #                 as_part=create_node.Name('d')
    #             )]
    #     )
    #     module_node = create_node.Module(node)
    #     string = 'with a as c, b as d:\n  pass\n'
    #     node.matcher = GetDynamicMatcher(node)
    #     node.matcher.do_match(string)
    #     node.body[0] = create_node.With(
    #         create_node.Name('e'),
    #         as_part=create_node.Name('f')
    #     )
    #     self.assertEqual('with a as c, e as f:\n  pass\n',
    #                      node.matcher.GetSource())


if __name__ == '__main__':
    unittest.main()
