from fun_with_ast.common_utils.node_tree_util import IsEmptyModule
from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node.create_node import GetNodeFromInput
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class BodyManipulator:

    def __init__(self, body_block):
        self.body_block = body_block
    def _add_newlines(self):
        for node in self.body_block:
            node_source = node.matcher.GetSource()
            if node_source.endswith("\n"):
                continue
            node.matcher.add_newline_to_source()

    def inject_node(self,node_to_inject, index):
        if IsEmptyModule(node_to_inject):
            return
        ident = self._get_indentation()
        source_to_inject = GetSource(node_to_inject, assume_no_indent=True)
        node_to_inject.matcher.FixIndentation(ident)
        self.body_block.insert(index, node_to_inject)
        self._add_newlines()

    def replace_body(self,source_of_new_body):
        body_lines = self._split_source_onto_lines(source_of_new_body)
        new_body = self._create_body_from_lines(body_lines)
        for node in new_body:
            matcher = GetDynamicMatcher(node)
            #self.node.matcher.orelse_placeholder._match(source_of_new_body)
            matcher.do_match(source_of_new_body)

        self.body_block = new_body
        return self.body_block

    def get_source(self):
        raise NotImplementedError('get_source not implemented yet')

    def _get_indentation(self):
        ident = 0
        for stmt in self.body_block:
            stmt_ident = stmt.col_offset
            if stmt_ident > ident and ident == 0:
                ident = stmt_ident
            elif stmt_ident != ident and ident != 0:
                raise ValueError('illegal ident')
        return ident

    def _split_source_onto_lines(self, source_of_new_body):
        body_lines = source_of_new_body.split('\n')
#        if len(body_lines) > 1:
#            raise NotImplementedError('replace_body not implemented yet')
        for body_line in body_lines:
            if body_line == '\n':
                raise NotImplementedError('Found end-of-line in if body')
        return body_lines

    def _create_body_from_lines(self, body_lines):
        new_body = []
        for line in body_lines:
            node = GetNodeFromInput(line.lstrip())
            new_body.append(node)
        return new_body
