import re

from fun_with_ast.common_utils.node_tree_util import IsEmptyModule
from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.manipulate_node.syntax_free_line_node import SyntaxFreeLine


class BodyManipulator:
    """ Class for manipulating the body of a node. (the If body, orelse body, etc.)"""
    def __init__(self, body_block):
        self.body_block = body_block
    def inject_node(self,node_to_inject, index):
        if IsEmptyModule(node_to_inject):
            return
        ident = self._get_indentation()

        source = GetSource(node_to_inject, assume_no_indent=True)
        # if not isinstance(node_to_inject, ast.Expr):
        #     node_to_inject.node_matcher.FixIndentation(ident)
        # else:
        #     node_to_inject.value.node_matcher.FixIndentation(ident)
        node_to_inject.node_matcher.FixIndentation(ident)
        self.body_block.insert(index, node_to_inject)
        self._add_newlines()

    def replace_body(self,source_of_new_body):
        if source_of_new_body == '':
            raise ValueError('source_of_new_body cannot be empty')
        new_body = self._create_body_from_source(source_of_new_body)
        module_node = create_node.Module(*new_body)
        self.body_block = module_node.body
        return self.body_block

    def get_source(self):
        raise NotImplementedError('get_source not implemented yet')

    def _add_newlines(self):
        for index, node in enumerate(self.body_block):
            node_source = node.node_matcher.GetSource()
            ends_with_new_line = re.search(r'\s*\n\s*$', node_source)
            if ends_with_new_line:
            #if node_source.endswith("\n"):
                continue
            if index == len(self.body_block) - 1:
                continue
            node.node_matcher.add_newline_to_source()


    def _get_indentation(self):
        ident = 0
        for stmt in self.body_block:
            if isinstance(stmt, SyntaxFreeLine):
                continue
            stmt_ident = stmt.col_offset
            if stmt_ident > ident and ident == 0:
                ident = stmt_ident
            elif stmt_ident != ident and ident != 0:
                raise ValueError('illegal ident')
        return ident

    def _create_body_from_source(self, body_lines):
        idented_body_source = self._ident_left(body_lines)
        new_body = GetNodeFromInput(idented_body_source, get_module=True).body
        return new_body

    def _ident_left(self, body_lines):
        body_lines = [x+'\n' for x in body_lines.split('\n') if x]
        new_body = []
        shift_count = self._get_ident_count_for_lines(body_lines)
        for line in body_lines:
            if line.startswith('#'):
                new_body.append(line)
            else:
                new_body.append(line[shift_count:])
        return ''.join(new_body)

    def _get_ident_count_for_lines(self, body_lines):
        result = 0
        for line in body_lines:
            line_ident = len(line) - len(line.lstrip())
            if line_ident == 0 and line.startswith('#'):
                continue
            result = line_ident
            break
        return result