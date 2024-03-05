from fun_with_ast.placeholders.composite import FieldPlaceholder
from fun_with_ast.placeholders.text import TextPlaceholder
from fun_with_ast.source_matchers.defualt_matcher import DefaultSourceMatcher


class LambdaSourceMatcher(DefaultSourceMatcher):
    def __init__(self, node, starting_parens=None, parent=None):
        expected_parts = [
                 TextPlaceholder(r'lambda\s*', 'lambda '),
                 FieldPlaceholder('args'),
                 #TextPlaceholder(r'\s*:\s*', ': '),
                 FieldPlaceholder('body'),
             ]

        super(LambdaSourceMatcher, self).__init__(node, expected_parts, starting_parens)
