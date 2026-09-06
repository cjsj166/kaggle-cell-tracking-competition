# Coding Style & Execution Constraints

- **Minimalist & Pragmatic**: Write only the lean, essential logic required to satisfy the immediate requirement.
- **No Speculative Error Handling**:
  - Do NOT wrap standard logic in defensive `try-catch` blocks unless explicitly requested or when handling uncontrollable external I/O (e.g., network calls).
  - Avoid defensive assertions, paranoid `None`/`null` validations, or arbitrary fallbacks for hypothetical edge cases.
  - Rely on the language's native exception-raising behavior; let unexpected failures raise directly.
- **Fail Fast & Loud**: When an error does occur, let the native runtime stack trace surface immediately rather than masking it behind custom exception handlers.
- **Happy Path Priority**: Focus solely on the correct execution path. Do not implement premature enterprise patterns, logging wrappers, or multi-tiered fallback cascades.