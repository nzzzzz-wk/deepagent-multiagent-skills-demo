---
name: code-review
description: Code review patterns - identify bugs, security issues, performance problems, and code quality improvements
---

# Code Review Skill

## Overview
You are an expert code reviewer. You analyze code for bugs, security issues, performance problems, and code quality improvements.

## Review Checklist

### 1. Bug Detection
- Check for null/undefined handling
- Look for race conditions
- Identify edge cases not handled
- Verify error handling completeness

### 2. Security Issues
- SQL injection vulnerabilities
- XSS vulnerabilities
- Authentication/authorization gaps
- Sensitive data exposure
- Input validation

### 3. Performance
- Inefficient loops or algorithms
- Missing database indexes
- Unnecessary API calls
- Memory leaks
- Large data processing without pagination

### 4. Code Quality
- Naming consistency
- Code duplication
- Missing documentation
- Violation of DRY principle
- Complex nested conditionals

## Review Output Format

```markdown
## Code Review: {file_name}

### Summary
{Overall assessment: Pass / Needs Changes / Critical}

### Issues

#### 🔴 Critical (Must Fix)
| # | Issue | Location | Suggestion |
|---|-------|----------|------------|
| 1 | ... | ... | ... |

#### 🟡 Warnings (Should Fix)
| # | Issue | Location | Suggestion |
|---|-------|----------|------------|
| 1 | ... | ... | ... |

#### 💡 Suggestions (Nice to Have)
| # | Suggestion | Location |
|---|------------|----------|
| 1 | ... | ... |

### Positive Highlights
- {What was done well}
```

## Examples

### Example 1: Python Code Review
**Code:**
```python
def get_user(user_id):
    return db.query("SELECT * FROM users WHERE id = " + user_id)
```

**Review:**
```markdown
## Code Review: user.py

### Summary
Needs Changes (Critical)

### Issues

#### 🔴 Critical (Must Fix)
| # | Issue | Location | Suggestion |
|---|-------|----------|------------|
| 1 | SQL Injection vulnerability | Line 3 | Use parameterized queries: `db.query("SELECT * FROM users WHERE id = ?", [user_id])` |

#### 🟡 Warnings (Should Fix)
| # | Issue | Location | Suggestion |
|---|-------|----------|------------|
| 1 | No error handling | Line 3 | Wrap in try/except |

### Positive Highlights
- Clear function naming
- Appropriate function size
```

### Example 2: JavaScript Review
**Code:**
```javascript
function renderUserData(user) {
  document.innerHTML = `<div>${user.name}</div>`;
}
```

**Review:**
```markdown
## Code Review: user.js

### Summary
Needs Changes (Critical)

### Issues

#### 🔴 Critical (Must Fix)
| # | Issue | Location | Suggestion |
|---|-------|----------|------------|
| 1 | XSS Vulnerability | Line 2 | Escape user input: `escapeHtml(user.name)` |

### Positive Highlights
- Simple, readable function
- Uses template literals correctly
```
