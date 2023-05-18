from fun_with_ast.placeholders.text_placeholder import TextPlaceholder

class DocStringTextPlaceholder(TextPlaceholder):
    """Placeholder for text (non-field). For example, 'def (' in FunctionDef."""

    def __init__(self):
        super(DocStringTextPlaceholder, self).__init__(r"([ \t]*'''.*'''[ \t]*\n)*", default='')

