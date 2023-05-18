
import re

from fun_with_ast.placeholders.text_placeholder import TextPlaceholder

from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.placeholders.base_placeholder import Placeholder


class WhiteSpaceTextPlaceholder(TextPlaceholder):
    """Placeholder for text (non-field). For example, 'def (' in FunctionDef."""

    def __init__(self):
        super(WhiteSpaceTextPlaceholder, self).__init__(r'[ \t]*', default='', longest_match=False)

