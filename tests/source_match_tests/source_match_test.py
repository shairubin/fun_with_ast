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


Tests for source_match.py
"""

import unittest

import pytest

import list_placeholder_source_match
from fun_with_ast.exceptions_source_match import BadlySpecifiedTemplateError

import defualt_source_matcher_source_match
import fun_with_ast.create_node as create_node
import source_match
from dynamic_matcher import GetDynamicMatcher

DEFAULT_TEXT = 'default'


class TextPlaceholderTest(unittest.TestCase):

    def testMatchSimpleText(self):
        placeholder = source_match.TextPlaceholder('.*', DEFAULT_TEXT)
        matched_text = placeholder.Match(None, 'to match')
        self.assertEqual(matched_text, 'to match')
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, 'to match')

    def testPartialMatchEnd(self):
        placeholder = source_match.TextPlaceholder(r'def \(', DEFAULT_TEXT)
        matched_text = placeholder.Match(None, 'def (foo')
        self.assertEqual(matched_text, 'def (')
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, 'def (')

    def testMatchWithoutMatchingReturnsDefault(self):
        placeholder = source_match.TextPlaceholder('.*', DEFAULT_TEXT)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, DEFAULT_TEXT)

    def testCantMatchThrowsError(self):
        placeholder = source_match.TextPlaceholder('doesnt match', DEFAULT_TEXT)
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder.Match(None, 'to match')

    def testMatchWhitespace(self):
        whitespace_text = '  \t \n  '
        placeholder = source_match.TextPlaceholder(r'\s*')
        matched_text = placeholder.Match(None, whitespace_text)
        self.assertEqual(matched_text, whitespace_text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, whitespace_text)

    def testWhitespaceMatchesLineContinuations(self):
        whitespace_text = '  \t \n \\\n  \\\n  '
        placeholder = source_match.TextPlaceholder(r'\s*')
        matched_text = placeholder.Match(None, whitespace_text)
        self.assertEqual(matched_text, whitespace_text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, whitespace_text)

    def testWhitespaceMatchesComments(self):
        whitespace_text = '  \t # abc\n  '
        placeholder = source_match.TextPlaceholder(r'\s*')
        matched_text = placeholder.Match(None, whitespace_text)
        self.assertEqual(matched_text, whitespace_text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, whitespace_text)

    def testMultipleStatementsSeparatedBySemicolon(self):
        whitespace_text = 'pdb;pdb'
        placeholder = source_match.TextPlaceholder(r'pdb\npdb')
        matched_text = placeholder.Match(None, whitespace_text)
        self.assertEqual(matched_text, whitespace_text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, whitespace_text)

    def testCommentAfterExpectedLinebreak(self):
        whitespace_text = 'pdb  #  \t A comment\n'
        placeholder = source_match.TextPlaceholder(r'pdb\n')
        matched_text = placeholder.Match(None, whitespace_text)
        self.assertEqual(matched_text, whitespace_text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, whitespace_text)

    def testCommentInNewLine(self):
        text = '\n   #  A comment\n'
        placeholder = source_match.TextPlaceholder('\n   #  A comment\n')
        matched_text = placeholder.Match(None, text)
        self.assertEqual(matched_text, text)
        test_output = placeholder.GetSource(None)
        self.assertEqual(test_output, text)


class FieldPlaceholderTest(unittest.TestCase):

    def testMatchSimpleFieldWithSpace(self):
        node = create_node.Name('foobar')
        placeholder = source_match.FieldPlaceholder('id')
        matched_text = placeholder.Match(node, 'foobar\t')
        self.assertEqual(matched_text, 'foobar')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar')
        matched_text = placeholder.Match(node, 'foobar\t\t\n')
        self.assertEqual(matched_text, 'foobar')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar')
        with self.assertRaises(BadlySpecifiedTemplateError):
            matched_text = placeholder.Match(node, ' foobar\t\t\n')
            self.assertEqual(matched_text, 'foobar')
            test_output = placeholder.GetSource(node)
            self.assertEqual(test_output, 'foobar')

    def testMatchSimpleField(self):
        node = create_node.Name('foobar')
        placeholder = source_match.FieldPlaceholder('id')
        matched_text = placeholder.Match(node, 'foobar')
        self.assertEqual(matched_text, 'foobar')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar')

    def testPartialMatch(self):
        node = create_node.Name('bar')
        placeholder = source_match.FieldPlaceholder(
            'id', before_placeholder=source_match.TextPlaceholder('foo'))
        matched_text = placeholder.Match(node, 'foobarbaz')
        self.assertEqual(matched_text, 'foobar')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar')

    def testBeforePlaceholder(self):
        node = create_node.Name('bar')
        placeholder = source_match.FieldPlaceholder(
            'id',
            before_placeholder=source_match.TextPlaceholder('before '))
        matched_text = placeholder.Match(node, 'before bar')
        self.assertEqual(matched_text, 'before bar')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'before bar')

    def testCantMatchThrowsError(self):
        node = create_node.Name('doesnt_match')
        placeholder = source_match.FieldPlaceholder('id')
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder.Match(node, 'to match')

    def testRaisesErrorIfFieldIsList(self):
        node = create_node.FunctionDef('function_name')
        placeholder = source_match.FieldPlaceholder('body')
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder.Match(node, 'invalid_match')

    def testChangingValueChangesOutput(self):
        node = create_node.Name('bar')
        placeholder = source_match.FieldPlaceholder(
            'id', before_placeholder=source_match.TextPlaceholder('foo'))
        matched_text = placeholder.Match(node, 'foobarbaz')
        self.assertEqual(matched_text, 'foobar')
        node.id = 'hello'
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foohello')

    def testWithoutMatch(self):
        node = create_node.Name('bar')
        placeholder = source_match.FieldPlaceholder('id')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'bar')


class ListFieldPlaceholderTest(unittest.TestCase):

    def testMatchSimpleField(self):
        body_node = create_node.Expr(create_node.Name('foobar'))
        node = create_node.FunctionDef('function_name', body=[body_node])
        placeholder = list_placeholder_source_match.ListFieldPlaceholder('body')
        matched_text = placeholder.Match(node, 'foobar\n')
        self.assertEqual(matched_text, 'foobar\n')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar\n')

    def testMultipleListItems(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        placeholder = list_placeholder_source_match.ListFieldPlaceholder('body')
        matched_text = placeholder.Match(node, 'foobar\nbaz\n')
        self.assertEqual(matched_text, 'foobar\nbaz\n')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar\nbaz\n')

    def testMultipleListItemsBeginningAndEnd(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        placeholder = list_placeholder_source_match.ListFieldPlaceholder(
            'body',
            before_placeholder=source_match.TextPlaceholder('z'),
            after_placeholder=source_match.TextPlaceholder('zz'))
        matched_text = placeholder.Match(node, 'zfoobar\nzzzbaz\nzz')
        self.assertEqual(matched_text, 'zfoobar\nzzzbaz\nzz')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'zfoobar\nzzzbaz\nzz')

    def testMatchRaisesErrorIfFieldIsNotList(self):
        node = create_node.Name('bar')
        placeholder = list_placeholder_source_match.ListFieldPlaceholder(
            'id', before_placeholder=source_match.TextPlaceholder('\n', '\n'),
            exclude_first_before=True)
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder.Match(node, 'foobar\nbaz')

    def testMatchRaisesErrorIfFieldDoesntMatch(self):
        body_node = create_node.Expr(create_node.Name('foobar'))
        node = create_node.FunctionDef('function_name', body=[body_node])
        placeholder = list_placeholder_source_match.ListFieldPlaceholder(
            'body', before_placeholder=source_match.TextPlaceholder('\n', '\n'),
            exclude_first_before=True)
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder.Match(node, 'no match here')

    def testMatchRaisesErrorIfSeparatorDoesntMatch(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        placeholder = list_placeholder_source_match.ListFieldPlaceholder(
            'body', before_placeholder=source_match.TextPlaceholder('\n', '\n'),
            exclude_first_before=True)
        with self.assertRaises(BadlySpecifiedTemplateError):
            placeholder.Match(node, 'foobarbaz')

    # TODO: Renabled this after adding indent information to matchers
    @unittest.expectedFailure
    def testListDefaults(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        module_node = create_node.Module(node)
        placeholder = list_placeholder_source_match.ListFieldPlaceholder(
            'body', before_placeholder=source_match.TextPlaceholder('', ', '),
            exclude_first_before=True)
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, '  foobar\n,   baz\n')

class SeparatedListFieldPlaceholderTest(unittest.TestCase):

    def testMatchSepertedListSingleElement(self):
        node = create_node.Assign('foo', 1)
        placeholder = source_match.SeparatedListFieldPlaceholder('targets',
                                                                 after__separator_placeholder=source_match.TextPlaceholder(r'[ \t]*=[ \t]*', '='))
        matched_text = placeholder.Match(node, 'foo=1')
        self.assertEqual(matched_text, 'foo=')
        placeholder = source_match.FieldPlaceholder('value')
        matched_text = placeholder.Match(node, '1')
        self.assertEqual(matched_text, '1')

    def testMatchSepertedListSingleElementWithWS(self):
        node = create_node.Assign('foo', 1)
        placeholder = source_match.SeparatedListFieldPlaceholder('targets',
                                                                 after__separator_placeholder=source_match.TextPlaceholder(r'[ \t]*=[ \t]*', '='))
        matched_text = placeholder.Match(node, 'foo \t   =\t  1')
        self.assertEqual(matched_text, 'foo \t   =\t  ')
        placeholder = source_match.FieldPlaceholder('value')
        matched_text = placeholder.Match(node, '1')
        self.assertEqual(matched_text, '1')

    def testMatchSepertedListSingleElementWithWSWithComment(self):
        node = create_node.Assign('foo', 1)
        placeholder = source_match.SeparatedListFieldPlaceholder('targets',
                                                                 after__separator_placeholder=source_match.TextPlaceholder(r'[ \t]*=[ \t]*', '='))
        matched_text = placeholder.Match(node, 'foo \t   =\t  1 # comment')
        self.assertEqual(matched_text, 'foo \t   =\t  ')
        placeholder = source_match.FieldPlaceholder('value')
        matched_text = placeholder.Match(node, '1')
        self.assertEqual(matched_text, '1')

    def testMatchSepertedList(self):
        node = create_node.Assign(['foo', 'bar'], 2)
        placeholder = source_match.SeparatedListFieldPlaceholder('targets',
                                                                 after__separator_placeholder=source_match.TextPlaceholder(r'[ \t]*=[ \t]*', '='))
        matched_text = placeholder.Match(node, 'foo=bar=2')
        self.assertEqual(matched_text, 'foo=bar=')
        placeholder = source_match.FieldPlaceholder('value')
        matched_text = placeholder.Match(node, '2')
        self.assertEqual(matched_text, '2')


class TestDefaultSourceMatcher(unittest.TestCase):

    def testInvalidExpectedPartsType(self):
        node = create_node.Name('bar')
        with self.assertRaises(ValueError):
            defualt_source_matcher_source_match.DefaultSourceMatcher(node, ['blah'])

    def testBasicTextMatch(self):
        matcher = defualt_source_matcher_source_match.DefaultSourceMatcher(
            None, [source_match.TextPlaceholder('blah', DEFAULT_TEXT)])
        matcher.Match('blah')
        self.assertEqual(matcher.GetSource(), 'blah')

    def testRaisesErrorIfNoTextMatch(self):
        matcher = defualt_source_matcher_source_match.DefaultSourceMatcher(
            None, [source_match.TextPlaceholder('blah', DEFAULT_TEXT)])
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.Match('bla')

    def testBasicFieldMatch(self):
        node = create_node.Name('bar')
        matcher = defualt_source_matcher_source_match.DefaultSourceMatcher(
            node, [source_match.FieldPlaceholder('id')])
        matcher.Match('bar')
        self.assertEqual(matcher.GetSource(), 'bar')

    def testRaisesErrorIfNoFieldMatch(self):
        node = create_node.Name('bar')
        matcher = defualt_source_matcher_source_match.DefaultSourceMatcher(
            node, [source_match.FieldPlaceholder('id')])
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.Match('ba')

    def testBasicFieldMatchWhenChangedFieldValue(self):
        node = create_node.Name('bar')
        matcher = defualt_source_matcher_source_match.DefaultSourceMatcher(
            node, [source_match.FieldPlaceholder('id')])
        matcher.Match('bar')
        node.id = 'foo'
        self.assertEqual(matcher.GetSource(), 'foo')

    def testBasicListMatch(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        matcher = defualt_source_matcher_source_match.DefaultSourceMatcher(
            node, [list_placeholder_source_match.ListFieldPlaceholder('body')])
        matcher.Match('foobar\nbaz\n')
        self.assertEqual(matcher.GetSource(), 'foobar\nbaz\n')

    def testRaisesErrorWhenNoMatchInBasicList(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        matcher = defualt_source_matcher_source_match.DefaultSourceMatcher(
            node, [list_placeholder_source_match.ListFieldPlaceholder('body')])
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.Match('foobar\nba\n')

    def testBasicListMatchWhenChangedFieldValue(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        matcher = defualt_source_matcher_source_match.DefaultSourceMatcher(
            node,
            [list_placeholder_source_match.ListFieldPlaceholder('body')])
        matcher.Match('foobar\nbaz\n')
        node.body[0].value.id = 'hello'
        self.assertEqual(matcher.GetSource(), 'hello\nbaz\n')

    def testAdvancedMatch(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        matcher = defualt_source_matcher_source_match.DefaultSourceMatcher(
            node,
            [source_match.TextPlaceholder('def ', 'def '),
             source_match.FieldPlaceholder('name'),
             source_match.TextPlaceholder(r'\(\)', r'()'),
             list_placeholder_source_match.ListFieldPlaceholder('body')])
        matcher.Match('def function_name()foobar\nbaz\n')
        node.body[0].value.id = 'hello'
        self.assertEqual(matcher.GetSource(), 'def function_name()hello\nbaz\n')

    # TODO: Renabled this after adding indent information to matchers
    @unittest.expectedFailure
    def testGetSourceWithoutMatchUsesDefaults(self):
        body_nodes = [create_node.Expr(create_node.Name('foobar')),
                      create_node.Expr(create_node.Name('baz'))]
        node = create_node.FunctionDef('function_name', body=body_nodes)
        module_node = create_node.Module(node)
        matcher = defualt_source_matcher_source_match.DefaultSourceMatcher(
            node,
            [source_match.TextPlaceholder('def ', 'default '),
             source_match.FieldPlaceholder('name'),
             source_match.TextPlaceholder(r'\(\)', r'()'),
             source_match.SeparatedListFieldPlaceholder(
                 'body', source_match.TextPlaceholder('\n', ', '))])
        node.body[0].value.id = 'hello'
        self.assertEqual(matcher.GetSource(),
                         'default function_name()  hello\n,   baz\n')


class TestGetMatcher(unittest.TestCase):

    def testDefaultMatcher(self):
        node = create_node.VarReference('foo', 'bar')
        matcher = GetDynamicMatcher(node)
        matcher.Match('foo.bar')
        self.assertEqual(matcher.GetSource(), 'foo.bar')

    def testDefaultMatcherWithModification(self):
        node = create_node.VarReference('foo', 'bar')
        matcher = GetDynamicMatcher(node)
        matcher.Match('foo.bar')
        node.attr = 'hello'
        self.assertEqual(matcher.GetSource(), 'foo.hello')





class AssertMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Assert(create_node.Name('a'))
        string = 'assert a\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithMessage(self):
        node = create_node.Assert(create_node.Name('a'),
                                  create_node.Str('message'))
        string = 'assert a, "message"\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


class AttributeMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.VarReference('a', 'b')
        string = 'a.b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testTripleReferenceMatch(self):
        node = create_node.VarReference('a', 'b', 'c')
        string = 'a.b.c'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


class AugAssignMatcherTest(unittest.TestCase):

    @pytest.mark.xfail(strict=True)
    def testBasicMatch(self):
        node = create_node.AugAssign('a', create_node.Add(), create_node.Num(12))
        string = 'a+=1\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testNotMatchWithVarAndTab(self):
        node = create_node.AugAssign('a', create_node.Add(), create_node.Name('c'))
        string = '       \t        a += b\n'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)
        #self.assertNotEqual(string, matcher.GetSource())

    def testMatchWithVarAndTab(self):
        node = create_node.AugAssign('a', create_node.Add(), create_node.Name('b'))
        string = '       \t        a += b\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithVarAndTab2(self):
        node = create_node.AugAssign('a', create_node.Add(), create_node.Name('b'))
        string = '               a +=\tb\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


class BoolOpMatcherTest(unittest.TestCase):

    def testAndBoolOp(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.And(),
            create_node.Name('b'))
        string = 'a and b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testOrBoolOp(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.Or(),
            create_node.Name('b'))
        string = 'a or b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testAndOrBoolOp(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.And(),
            create_node.Name('b'),
            create_node.Or(),
            create_node.Name('c'))
        string = 'a and b or c'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testOrAndBoolOp(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.Or(),
            create_node.Name('b'),
            create_node.And(),
            create_node.Name('c'))
        string = 'a or b and c'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


class ClassDefMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.ClassDef('TestClass', body=[create_node.Pass()])
        string = 'class TestClass():\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchBases(self):
        node = create_node.ClassDef(
            'TestClass', bases=['Base1', 'Base2'], body=[create_node.Pass()])
        string = 'class TestClass(Base1, Base2):\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchBody(self):
        node = create_node.ClassDef(
            'TestClass', body=[create_node.Expr(create_node.Name('a'))])
        string = 'class TestClass():\n  a\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchDecoratorList(self):
        node = create_node.ClassDef(
            'TestClass',
            decorator_list=[create_node.Name('dec'),
                            create_node.Call('dec2')], body=[create_node.Pass()])
        string = '@dec\n@dec2()\nclass TestClass():\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
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
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testCanChangeValues(self):
        node = create_node.ClassDef(
            'TestClass',
            bases=['Base1', 'Base2'],
            body=[create_node.Expr(create_node.Name('a'))],
            decorator_list=[create_node.Name('dec'),
                            create_node.Call('dec2')])
        string = '@dec\n@dec2()\nclass TestClass(Base1, Base2):\n  a\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        node.bases = [create_node.Name('Base3')]
        node.decorator_list = [create_node.Name('dec3')]
        node.body[0].value.id = 'x'
        node.name = 'TestClass2'
        changed_string = '@dec3\nclass TestClass2(Base3):\n  x\n'
        self.assertEqual(changed_string, matcher.GetSource())


class CompareMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Lt(),
            create_node.Name('b'))
        string = 'a < b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMultiMatch(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Lt(),
            create_node.Name('b'),
            create_node.Lt(),
            create_node.Name('c'))
        string = 'a < b < c'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testEq(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Eq(),
            create_node.Name('b'))
        string = 'a == b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testNotEq(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.NotEq(),
            create_node.Name('b'))
        string = 'a != b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testLt(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Lt(),
            create_node.Name('b'))
        string = 'a < b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testLtE(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.LtE(),
            create_node.Name('b'))
        string = 'a <= b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testGt(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Gt(),
            create_node.Name('b'))
        string = 'a > b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testGtE(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.GtE(),
            create_node.Name('b'))
        string = 'a >= b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testIs(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.Is(),
            create_node.Name('b'))
        string = 'a is b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testIsNot(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.IsNot(),
            create_node.Name('b'))
        string = 'a is not b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testIn(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.In(),
            create_node.Name('b'))
        string = 'a in b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testNotIn(self):
        node = create_node.Compare(
            create_node.Name('a'),
            create_node.NotIn(),
            create_node.Name('b'))
        string = 'a not in b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


class ComprehensionMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.comprehension('a', 'b', False)
        string = 'for a in b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithIf(self):
        node = create_node.comprehension(
            'a', 'b', True,
            create_node.Compare('c', '<', 'd'))
        string = 'for a in b if c < d'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


class DictMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Dict([create_node.Name('a')],
                                [create_node.Name('b')])
        string = '{a: b}'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testEmptyMatch(self):
        node = create_node.Dict()
        string = '{}'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testTwoItemMatch(self):
        node = create_node.Dict(
            [create_node.Name('a'), create_node.Str('c')],
            [create_node.Name('b'), create_node.Str('d')])
        string = '{a: b, "c": "d"}'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testChangeKey(self):
        first_key = create_node.Name('a')
        node = create_node.Dict(
            [first_key, create_node.Str('c')],
            [create_node.Name('b'), create_node.Str('d')])
        string = '{a: b, "c": "d"}'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        first_key.id = 'k'
        self.assertEqual('{k: b, "c": "d"}', matcher.GetSource())

    def testChangeVal(self):
        first_val = create_node.Name('b')
        node = create_node.Dict(
            [create_node.Name('a'), create_node.Str('c')],
            [first_val, create_node.Str('d')])
        string = '{a: b, "c": "d"}'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        first_val.id = 'k'
        self.assertEqual('{a: k, "c": "d"}', matcher.GetSource())


class DictComprehensionMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.DictComp('e', 'f', 'a', 'b')
        string = '{e: f for a in b}'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithIf(self):
        node = create_node.DictComp(
            'e', 'f', 'a', 'b',
            create_node.Compare('c', '<', 'd'))
        string = '{e: f for a in b if c < d}'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


class ExceptHandlerMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.ExceptHandler()
        string = 'except:\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithType(self):
        node = create_node.ExceptHandler('TestException')
        string = 'except TestException:\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithName(self):
        node = create_node.ExceptHandler('TestException', name='as_part')
        string = 'except TestException as as_part:\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithBody(self):
        node = create_node.ExceptHandler(
            body=[create_node.Expr(create_node.Name('a'))])
        string = 'except:\n  a\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


class IfExpMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.IfExp(
            create_node.Name('True'), create_node.Name('a'), create_node.Name('b'))
        string = 'a if True else b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    @pytest.mark.xfail(strict=True)
    def testChangeParts(self):
        node = create_node.IfExp(
            create_node.Name('True'), create_node.Name('a'), create_node.Name('b'))
        string = 'a if True else b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        node.test = create_node.Name('False')
        node.body = create_node.Name('c')
        node.orelse = create_node.Name('d')
        self.assertEqual('c if False else d', matcher.GetSource())


class ListComprehensionMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.ListComp('c', 'a', 'b')
        string = '[c for a in b]'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithIf(self):
        node = create_node.ListComp(
            'c', 'a', 'b',
            create_node.Compare('c', '<', 'd'))
        string = '[c for a in b if c < d]'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


class SetComprehensionMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.SetComp('c', 'a', 'b')
        string = '{c for a in b}'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithIf(self):
        node = create_node.SetComp(
            'c', 'a', 'b',
            create_node.Compare('c', '<', 'd'))
        string = '{c for a in b if c < d}'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


class StrMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Str('foobar')
        string = '"foobar"'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual('"foobar"', matcher.GetSource())

    def testPrefixMatch(self):
        node = create_node.Str('foobar')
        string = 'r"foobar"'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual('r"foobar"', matcher.GetSource())

    def testQuoteWrapped(self):
        node = create_node.Str('foobar')
        string = '("foobar")'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual('("foobar")', matcher.GetSource())

    def testContinuationMatch(self):
        node = create_node.Str('foobar')
        string = '"foo"\n"bar"'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(NotImplementedError):
            matcher.Match(string)

    def testContinuationMatchWithPrefix(self):
        node = create_node.Str('foobar')
        string = '"foo"\nr"bar"'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(NotImplementedError):
            matcher.Match(string)

    def testBasicTripleQuoteMatch(self):
        node = create_node.Str('foobar')
        string = '"""foobar"""'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual('"""foobar"""', matcher.GetSource())

    def testMultilineTripleQuoteMatch(self):
        node = create_node.Str('foobar\n\nbaz')
        string = '"""foobar\n\nbaz"""'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual('"""foobar\n\nbaz"""', matcher.GetSource())

    def testQuoteTypeMismatch(self):
        node = create_node.Str('foobar')
        string = '"foobar\''
        matcher = GetDynamicMatcher(node)
        with self.assertRaises(ValueError):
            matcher.Match(string)

    def testSChange(self):
        node = create_node.Str('foobar')
        string = '"foobar"'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        node.s = 'hello'
        self.assertEqual('"hello"', matcher.GetSource())

    def testSChangeInContinuation(self):
        node = create_node.Str('foobar')
        string = '"foo"\n"bar"'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(NotImplementedError):
            matcher.Match(string)
        #node.s = 'foobaz'
        #self.assertEqual('"foobaz"', matcher.GetSource())

    def testQuoteTypeChange(self):
        node = create_node.Str('foobar')
        string = '"foobar"'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher.str_matcher.quote_type = "'"
        self.assertEqual("'foobar'", matcher.GetSource())

    def testQuoteTypeChangeToTripleQuote(self):
        node = create_node.Str('foobar')
        string = '"foobar"'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher.str_matcher.quote_type = "'''"
        matched_source = matcher.GetSource()
        self.assertEqual("'''foobar'''", matched_source)


class CommentMatcherTest(unittest.TestCase):
    def testBasicMatch(self):
        node = create_node.Comment('#comment')
        string = '#comment'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchSpecialChars(self):
        string = '##\t#   c \to m \t  m e n t ? # $'
        node = create_node.Comment(string)
        matcher = GetDynamicMatcher(node)
        matched_text = matcher.Match(string)
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
    #    matcher.Match(string)
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
        matcher.Match(string)
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
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


class WithItemMatcherTest(unittest.TestCase):

    def testBasicWithItem(self):
        node = create_node.withitem('a')
        string = 'a'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testWithItemWithAs(self):
        node = create_node.withitem('a', optional_vars='b')
        string = 'a    as     b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)


class WithMatcherTest(unittest.TestCase):
    # start here next time
    def testBasicWith(self):
        node = create_node.With(
            [create_node.withitem('a')], [create_node.Pass()])
        string = 'with a:\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicWithAs(self):
        node = create_node.With([create_node.withitem('a', optional_vars='b')], [create_node.Pass()])
        string = 'with a as b:\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testWithAsTuple(self):
        node = create_node.With([create_node.withitem('a', optional_vars=create_node.Tuple(['b', 'c']))],
                                [create_node.Pass()])
        string = 'with   a  as     (b,  c):  \n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    # not relevant when using withitem
    # def testChangeWithAsTuple(self):
    #     node = create_node.With([create_node.withitem('a', optional_vars=create_node.Tuple(['b', 'c']))],
    #                             [create_node.Pass()])
    #     string = 'with a as (b, c):\n  pass\n'
    #     matcher = GetDynamicMatcher(node)
    #     matcher.Match(string)
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
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

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
    #     node.matcher.Match(string)
    #     node.body[0] = create_node.With(
    #         create_node.Name('e'),
    #         as_part=create_node.Name('f')
    #     )
    #     self.assertEqual('with a as c, e as f:\n  pass\n',
    #                      node.matcher.GetSource())


if __name__ == '__main__':
    unittest.main()
