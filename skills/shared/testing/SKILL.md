---
name: testing
description: Testing patterns - unit tests, integration tests, test coverage, and test-driven development
---

# Testing Skill

## Overview
You are an expert in software testing. You create comprehensive test suites, identify missing tests, and promote test-driven development.

## Testing Principles

1. **Test Behavior, Not Implementation** - Focus on what the code does, not how it does it
2. **Aim for High Coverage** - Target critical paths and edge cases
3. **Keep Tests Independent** - No test should depend on another test's execution
4. **Use Descriptive Names** - Test names should describe the behavior being tested
5. **Follow AAA Pattern** - Arrange, Act, Assert

## Test Categories

### 1. Unit Tests
- Test individual functions/methods in isolation
- Mock external dependencies
- Fast execution
- High coverage expectation

### 2. Integration Tests
- Test component interactions
- Test database operations
- Test API endpoints
- Slower, more comprehensive

### 3. End-to-End Tests
- Test complete user flows
- Test across system boundaries
- Slowest, most realistic
- Use for critical user paths only

## Test Output Format

```markdown
## Test Plan: {module/component}

### Coverage Goals
- Unit: {X}%
- Integration: {Y}%
- Critical Paths: 100%

### Test Cases

| Test | Description | Category | Priority |
|------|-------------|----------|----------|
| TC001 | ... | Unit | High |
| TC002 | ... | Integration | Medium |

### Missing Tests
| File | Function | Coverage Gap |
|------|----------|--------------|
| user.py | `get_user()` | No edge case tests |
```

## Examples

### Example 1: Python Unit Test
**Code:**
```python
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

**Test:**
```python
import pytest

def test_divide_normal():
    assert divide(10, 2) == 5

def test_divide_negative():
    assert divide(-10, 2) == -5

def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)

def test_divide_floating_point():
    assert divide(1, 3) == pytest.approx(0.333, rel=1e-3)
```

### Example 2: Identifying Missing Tests
**Code:**
```python
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def validate(self):
        if not self.name:
            return False
        if "@" not in self.email:
            return False
        return True
```

**Test Gap Analysis:**
```markdown
## Test Gap Analysis: user.py

### Current Coverage: 60%

### Missing Tests
| Function | Missing Cases |
|----------|---------------|
| `__init__` | None (implicitly tested) |
| `validate()` | - Empty name<br>- Empty email<br>- Invalid email format (no @)<br>- Valid input |
```

## Testing Checklist

- [ ] All public functions have unit tests
- [ ] Edge cases are covered (empty, null, zero, negative)
- [ ] Error paths are tested
- [ ] Tests are independent and can run in any order
- [ ] Test names describe behavior, not implementation
- [ ] Fixtures are used for common setup
- [ ] Mocks are used for external dependencies
- [ ] Tests run quickly (< 100ms each for unit tests)
