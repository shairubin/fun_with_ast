import ast
import sys

import pytest

from fun_with_ast.common_utils.node_tree_util import IsEmptyModule
from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node.body_manipulator import BodyManipulator
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.manipulate_node.if_manipulator import ManipulateIfNode, IfManipulatorConfig
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.manipulate_tests.base_test_utils_manipulate import bcolors

input_legend = ('inject-source', 'location', 'original-if', 'expected', 'match-expected', 'injected_second_source')


@pytest.fixture(params=[
    ('a.b()\n', 0, 'if (c.d()):\n   a=1\n', 'if (c.d()):\n   a.b()\n   a=1\n', True, 'b.a()'),  # 0
    ('a.c()\n', 0, 'if (c.d()):\n   a=1\n', 'if (c.d()):\n   a.c()\n   a=1\n', True, 'print(test)'),  # 1
    ('a=44\n', 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=44\n   a=1\n', True, 'pass # comment'),  # 2
    ("s='fun_with_ast'\n", 0, 'if (c.d()):\n   a=1\n', "if (c.d()):\n   s='fun_with_ast'\n   a=1\n", True, 'raise(test)'),
    # 3
    ("", 0, 'if (c.d()):\n   a=1\n', 'if (c.d()):\n   a=1\n', True, 'a.x()'),  # 4
    ('a.b()\n', 1, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=1\n   a.b()\n', True, 'False'),  # 5
    ('a.c()\n', 1, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=1\n   a.c()\n', True, '   # only comment'),  # 6
    ("", 1, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=1\n', True, 'pass'),  # 7
    ('a.bb()\n', 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a.b()\n   a=1\n', False, 'a.b.c'),  # 8
    ('a.c()\n', 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a.b()\n   a=1\n', False, 'a=99'),  # 9
    ('a=45\n', 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=44\n\n   a=1\n', False, ''),
    ("s='fun_with_ast2'\n", 0, 'if (c.d()):\n   a=1', "if (c.d()):\n   s='fun_with_ast2'\n   a=1\n", True, 'raise'),
    ("", 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n    a=1\n', False, 'a<b'),  # 12
    ('a.b()\n', 1, 'if (c.d()):\n   a=1', 'if (c.x()):\n   a=1\n   a.b()\n', False, 'pass'),  # 13
    ('a.c()\n', 1, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=1\n   a.b()\n', False, 'pass'),  # 14
    ("", 1, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=2', False, ''),  # 15
    ('a.b()\n', 0, 'if (c.d()):\n #comment-line\n   a=1',  # 16
     'if (c.d()):\n   a.b()\n #comment-line\n   a=1\n', True, '   # another comment'),
    ('a.b()\n', 1, 'if (c.d()):\n #comment-line\n   a=1',  # 17
     'if (c.d()):\n #comment-line\n   a.b()\n   a=1\n', True, 'pass'),
    ('a.b()\n', 1, 'if (c.d()):\n #comment-line\n   a=1',  # 18
     'if (c.d()):\n #comment----line\n   a.b()\n   a=1\n', False, 'pass'),
    ('a.b()\n', 0, 'if (c.d()):\n\n   a=1',  # 19
     'if (c.d()):\n   a.b()\n\n   a=1\n', True, 'pass'),  # TODO: this is currently a weird behavior in which
    # empty line is counted as a line
    ('a.b()\n', 1, 'if (c.d()):\n\n   a=1',  # 20
     'if (c.d()):\n\n   a.b()\n   a=1\n', True, 'pass'),  # TODO: this is currently a weird behavior in
    # which empty line is counted as a line #24
    ('a.b()\n', 0, 'if (c.d()):\n   a=1\n # comment',  # 21
     'if (c.d()):\n   a.b()\n   a=1\n # comment\n', True, 'pass'),
    ('a.b()\n', 0, 'if (c.d()):\n   a=1\n   b=1',  # 22
     'if (c.d()):\n   a.b()\n   a=1\n   b=1\n', True, 'pass'),
    ('a.b()\n', 0, 'if (c.d()):\n   if True:\n      a=111\n   b=11\n   c=12\n',  # 23
    'if (c.d()):\n   a.b()\n   if True:\n      a=111\n   b=11\n   c=12\n', True, 'pass'),
    ('a.b()\n', 0, """if _utils.is_sparse(A):
        if len(A.shape) != 2:
            raise ValueError("pca_lowrank input is expected to be 2-dimensional tensor")
        c = torch.sparse.sum(A, dim=(-2,)) / m
""",  # 24
     """if _utils.is_sparse(A):
        a.b()
        if len(A.shape) != 2:
            raise ValueError("pca_lowrank input is expected to be 2-dimensional tensor")
        c = torch.sparse.sum(A, dim=(-2,)) / m
"""
     , True, 'b.a()'),

    ('a.b()\n', 0, """if True:
            if False:
                raise ValueError("test")
            c = a
""",  # 25
                """if True:
            a.b()
            if False:
                raise ValueError("test")
            c = a
""", True, ''),

    ('a.b()\n', 0, """if True:
        if False:
            a=1
        c = a
""",  # 26
     """if True:
        a.b()
        if False:
            a=1
        c = a
""" , True, ''),

('a.b()\n', 0, """if first_card == 100:
        self.direction = -1
        self.can_add_card = self.can_add_card_down
""",  # 27
     """if first_card == 100:
        a.b()
        self.direction = -1
        self.can_add_card = self.can_add_card_down
""" , True, ''),
('a.b()\n', 0, """if first_card == 100:
        self.direction = -1
        self.can_add_card = self.can_add_card_down
""",  # 28
     """if first_card == 100:
        a.b()
        self.direction = -1
        self.can_add_card = self.can_add_card_down)
""" , False, ''),

# ('a.b()\n', 0, """def argument():
#     should_default = not is_out
#     if isinstance(a, Argument):
#         if True:
#             pass
#             return []
#     elif False:
#         if should_default:
#             default = "{}"
# """,  # 29
#      """if first_card == 100:
#         a.b()
#         self.direction = -1
#         self.can_add_card = self.can_add_card_down
# """ , True, ''),

('a.b()\n', 0, """if isinstance(a, Argument):
        if True:
            pass
            a=1
elif False:
        if should_default:
            default = "{}"
""",  # 29
     """if isinstance(a, Argument):
        a.b()
        if True:
            pass
            a=1
elif False:
        if should_default:
            default = "{}"
""" , True, ''),

])
def injected_source(request):
    yield request.param


@pytest.fixture(scope="function", params=[
    #{"body": '   if True:\n      pass\n   a=88', "else_body": '', "inject_to": 0, "condition": 'c.d():'},

    {"body": '   pass\n', "else_body": '', "inject_to": 0, "condition": 'c.d():'},
    {"body": '   pass\n   z=x\n', "else_body": '   a=1\n', "inject_to": 0, "condition": 'c.d():'},
    {"body": '   a=1\n', "else_body": '   pass', "inject_to": 0, "condition": 'c.d(): # comment'},
    {
        "body": "  if x%2 == 0:\n    print(\"x is a positive even number\")\n  else:\n    print(\"x is a positive odd number\")\n",
        "else_body": '  a=1', "inject_to": 0, "condition": 'a>2: #comment'},
    {"body": '#line comment \n   pass #comment 1\n', "else_body": '   a=1 #comment 2',
     "inject_to": 0, "condition": 'a and not b and not not c:'},
    {
        "body": "  if x%2 == 0:\n    print(\"x is a positive even number\")\n  else:\n    print(\"x is a positive odd number\")\n",
        "else_body": '  a=1',
        "inject_to": 1, "condition": '(a and not b) or not (not c):'},
    {"body": '#line comment \n   pass #comment 1\n', "else_body": '   a=1 #comment 2', "inject_to": 1,
     "condition": 'a+b > c/d+c:'},
    {"body": '#line comment \n   pass #comment 1\n', "else_body": '   a=1 #comment 2', "inject_to": 0,
     "condition": 'a+b > c/(d+c):'}

])
def body_and_orelse(request):
    yield request.param


def _get_tuple_as_dict(in_tuple):
    return dict(zip(input_legend, in_tuple))


# @pytest.mark.usefixtures(body_and_orelse)
class TestIfManupulation:

    def test_If_Manipulation(self, injected_source, capsys):
        dict_input = _get_tuple_as_dict(injected_source)
        parsed_node = ast.parse(dict_input['original-if'])
        if_node, injected_node = self._create_nodes(capsys, dict_input['inject-source'], dict_input['original-if'],
                                                    '')
        manipulator = ManipulateIfNode(if_node,
                                       IfManipulatorConfig(body_index=0, location_in_body_index=dict_input['location']))
        manipulator.add_nodes(injected_node.body)
        composed_source = self._source_after_composition(if_node, capsys)
        if dict_input['match-expected']:
            assert composed_source == dict_input['expected']
        else:
            assert dict_input['expected'] != composed_source

    def test_If_Else_Manipulation(self, injected_source, capsys):
        original_if_source = 'if ( c.d() ):\n   a=1\nelse:\n   b=2'
        if_node, injected_node = self._create_nodes(capsys, injected_source[0], original_if_source)
        manipulator = ManipulateIfNode(if_node, IfManipulatorConfig(1, 1))
        manipulator.add_nodes(injected_node.body)
        composed_source = self._source_after_composition(if_node, capsys)

        add_new_line = '\n' if not injected_source[0].endswith('\n') else ''
        if not IsEmptyModule([injected_node]):
            expected_source = original_if_source.replace('b=2', 'b=2\n   ' + injected_source[0] + add_new_line)
        else:
            expected_source = original_if_source + '\n'
        assert composed_source == expected_source

    def test_Module_Body_Manipulation(self, injected_source, capsys):
        original_module_source = 'a=1\na=2\nif c.d():\n   b=1\n   b=2\nelse:\n   c=1\n   c=2'
        for index in [0,1,2,3]:
            module_node, injected_node = self._create_nodes(capsys, injected_source[0], original_module_source,
                                                        is_module=True)
            manipulator = BodyManipulator(module_node.body)
            manipulator.inject_node([injected_node], index)
            print("\n insert in index:" + str(index))
            composed_source = self._source_after_composition(module_node, capsys)
            composed_source_lines = composed_source.split('\n')
            if index in [0,1,2] and injected_source[0]:
                assert composed_source_lines[index] + '\n' == injected_source[0]
            if index ==  3 and injected_source[0]:
                assert composed_source_lines[-2] + '\n' == injected_source[0]
        return

    def test_If_elif_AddNode(self, injected_source, capsys):
        original_if_source = 'if ( c.d() ):\n   a=1\nelif e==2:\n   b=2'
        if_node, injected_node = self._create_nodes(capsys, injected_source[0], original_if_source)
        manipulator = ManipulateIfNode(if_node, IfManipulatorConfig(1, 1))
        manipulator.add_nodes(injected_node.body)
        composed_source = self._source_after_composition(if_node, capsys)
        add_new_line = '\n' if not injected_source[0].endswith('\n') else ''
        if not IsEmptyModule([injected_node]):
            expected_source = original_if_source.replace('b=2', 'b=2\n   ' + injected_source[0] + add_new_line)
        else:
            expected_source = original_if_source + '\n'
        assert composed_source == expected_source

    def test_get_source_body(self, body_and_orelse, capsys):
        body, body_index, orelse, test = self._get_test_pastameters(body_and_orelse)
        else_string = 'else:\n' if orelse else ''
        original_if_source = 'if ' + test + '\n' + body + else_string + orelse
        if_node, injected_node = self._create_nodes(capsys, 'pass', original_if_source)
        manipulator = ManipulateIfNode(if_node, IfManipulatorConfig(body_index, 1))
        the_source = manipulator.get_body_orelse_source()
        title = 'Body source:' if body_index == 0 else 'Else source:'
        title = 'test_get_source. ' + title
        self._capture_source(capsys, the_source, title, bcolors.OKGREEN, True)
        if body_index == 0:
            assert the_source == body
        elif body_index == 1:
            assert the_source == orelse
        else:
            raise ValueError("body index can be only 0 or 1")

    def test_If_Manipulation_multiple_intections(self, injected_source, capsys):
        dict_input = _get_tuple_as_dict(injected_source)
        parsed_node = ast.parse(dict_input['original-if'])
        if_node, injected_node = self._create_nodes(capsys, dict_input['inject-source'], dict_input['original-if'],
                                                            dict_input['injected_second_source']  )
        manipulator = ManipulateIfNode(if_node,
                                       IfManipulatorConfig(body_index=0, location_in_body_index=dict_input['location']))
        manipulator.add_nodes(injected_node.body)
        composed_source = self._source_after_composition(if_node, capsys)
        if dict_input['match-expected']:
            self._validate_multi_lines_added(dict_input, composed_source)
        else:
            assert dict_input['expected'] != composed_source

    def test_switch_body_else(self, body_and_orelse, capsys):
        body, body_index, orelse, test = self._get_test_pastameters(body_and_orelse)
        if not orelse:
            return  # nothing to do as there is no else body
        else_string = 'else:\n' if orelse else ''
        original_if_source = 'if ' + test + '\n' + body + else_string + orelse
        if_node, _ = self._create_nodes(capsys, 'pass', original_if_source)
        config = IfManipulatorConfig(0, 1)
        manipulator = ManipulateIfNode(if_node, config)
        body_source = manipulator.get_body_orelse_source()
        self._capture_source(capsys, body_source, 'Original Body source:', bcolors.OKGREEN, True)
        config.body_index = 1
        orig_else_source = manipulator.get_body_orelse_source()
        self._capture_source(capsys, orig_else_source, 'Original Else Source', bcolors.OKGREEN, True)
        assert body_source == body
        assert orig_else_source == orelse
        manipulator.rerplace_body(body_source)
        new_else_source = manipulator.get_body_orelse_source()
        self._capture_source(capsys, new_else_source, 'New Else Source', bcolors.OKCYAN, True)
        assert new_else_source == body_source
        assert new_else_source != orelse
        config.body_index = 0
        manipulator.rerplace_body(orig_else_source)
        new_body_source = manipulator.get_body_orelse_source()
        self._capture_source(capsys, new_body_source, 'New Body Source', bcolors.OKCYAN, True)
        if orig_else_source.endswith('\n'):
            add_new_line_to_new_body = ''
        else:
            add_new_line_to_new_body = '\n'
        assert new_body_source == orig_else_source + add_new_line_to_new_body
        expected_new_if_source = 'if ' + test + '\n' + orelse + add_new_line_to_new_body + 'else:\n' + body
        actual_new_if_source = if_node.node_matcher.GetSource()
        self._capture_source(capsys, actual_new_if_source, 'New If Source', bcolors.OKCYAN, True)
        assert expected_new_if_source == actual_new_if_source

    def _create_nodes(self, capsys, injected_source, original_source, injected_second_source='', is_module=False):
        self._capture_source(capsys, original_source, 'original source:', bcolors.OKBLUE)
        if_node = self._create_if_node(original_source, is_module)
        injected_node, injected_node_source = self._create_injected_node(injected_source, injected_second_source)
        return if_node, injected_node

    def _create_injected_node(self, injected_source, injected_second_source):
        injected_node_source = injected_source
        #module = False
        #if injected_second_source:
        injected_node_source +=  injected_second_source
        #    module=True
        injected_node = GetNodeFromInput(injected_node_source, get_module=True)
        injected_node_matcher = GetDynamicMatcher(injected_node)
        injected_node_matcher.do_match(injected_node_source)
        source_from_matcher = injected_node_matcher.GetSource()
        assert source_from_matcher == injected_node_source
        source_from_get_source = GetSource(injected_node, assume_no_indent=True)
        assert source_from_get_source == injected_node_source
        return injected_node, injected_node_source

    def _create_if_node(self, original_source, is_module):
        node = GetNodeFromInput(original_source, get_module=is_module)
        node_matcher = GetDynamicMatcher(node)
        node_matcher.do_match(original_source)
        if_node_source = node_matcher.GetSource()
        assert if_node_source == original_source
        # if_node.matcher = if_node_matcher
        return node

    def _capture_source(self, capsys, source, title, color, ignore_ident=False):
        if not ignore_ident:
            try:
                compile(source, '<string>', mode='exec')
            except Exception as e:
                assert False, f'Error in source compilation'
        print(color + '\n' + title + '\n' + source + bcolors.ENDC)
        out, _ = capsys.readouterr()
        sys.stdout.write(out)

    def _source_after_composition(self, if_node, capsys):
        composed_source = GetSource(if_node, assume_no_indent=True)
        self._capture_source(capsys, composed_source, 'Modified source', bcolors.OKCYAN)
        return composed_source

    def _get_test_pastameters(self, body_and_orelse):
        body = body_and_orelse['body']
        orelse = body_and_orelse['else_body']
        body_index = body_and_orelse['inject_to']
        test = body_and_orelse['condition']
        return body, body_index, orelse, test

    def _validate_multi_lines_added(self, dict_input, composed_source):
        #if dict_input['location'] != 0:
        #    raise ValueError('Only location 0 is supported for multiple lines')
        added_lines = []
        lines = composed_source.split('\n')
        for index , line in enumerate(lines):
            if dict_input['inject-source'] != '' and dict_input['inject-source'] in line +'\n':
                next_added_line = 1
                if dict_input['injected_second_source'] and dict_input['injected_second_source'] in lines[index+1]+'\n':
                    self._validate_manipilation_multi_lines(added_lines, dict_input, index, lines,
                                                            next_added_line)
                    return True
                elif dict_input['injected_second_source'] == '':
                    next_added_line = 0
                    self._validate_manipilation_multi_lines(added_lines, dict_input, index, lines,
                                                            next_added_line)
                    return True
                else:
                    raise ValueError('not implemented yet')
            elif dict_input['inject-source'] == '' and dict_input['injected_second_source'] in line +'\n':
                next_added_line = 0
                self._validate_manipilation_multi_lines(added_lines, dict_input, index, lines,
                                                        next_added_line)
                return True
        raise ValueError('multi_lines failed')

    def _validate_manipilation_multi_lines(self, added_lines, dict_input, index, lines, next_added_line):
        added_lines.extend([index, index + next_added_line])
        self._validate_original_source(lines.copy(), added_lines, dict_input)
        self._validate_same_indentation(lines[index], lines[index + next_added_line])

    def _validate_same_indentation(self, line1, line2):
        count1 = len(line1) - len(line1.lstrip())
        count2 = len(line2) - len(line2.lstrip())
        assert count1 == count2, 'Indentation is not the same for the two lines'

    def _validate_original_source(self, lines, added_lines, dict_input):
        del lines[added_lines[0]:added_lines[1]+1]
        original_source = '\n'.join(lines)
        if dict_input['original-if'].endswith('\n'):
            assert original_source == dict_input['original-if'], 'Original source is not as expected'
        else:
            assert original_source  == dict_input['original-if'] + '\n', 'Original source is not as expected'