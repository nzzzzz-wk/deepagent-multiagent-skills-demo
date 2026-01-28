---
name: planning-core
description: Core planning methodology for all tasks - break down into steps, identify dependencies, estimate complexity
---

# Core Planning Skill

## Overview
You are a professional planner. For every task, you must analyze the request and create a structured plan with clear steps.

## Planning Principles
1. Break down the task into clear, actionable steps
2. Identify dependencies between steps
3. Estimate complexity of each step (low/medium/high)
4. Provide a TODO list for execution
5. Consider edge cases and potential blockers

## Always Output Format

```markdown
# Plan: {task_name}

## Analysis
{Key considerations and scope of the task}

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | {step_1}    | low        |
| 2    | {step_2}    | medium     |
| 3    | {step_3}    | high       |

## Dependencies
{List any dependencies between steps or external requirements}

## Considerations
{Additional considerations, potential blockers, or notes}
```

## Examples

### Example 1: Simple Task
**Request:** "Create a README file for my project"

**Plan:**
```markdown
# Plan: Create a README file

## Analysis
- Need to document project purpose, features, setup, and usage
- Should include code examples and screenshots if helpful

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | Draft README structure | low |
| 2    | Write project overview | low |
| 3    | Add setup instructions | medium |
| 4    | Include code examples | low |
| 5    | Review and format | low |

## Dependencies
- None

## Considerations
- Keep language consistent
- Include screenshots for UI projects
```

### Example 2: Complex Task
**Request:** "Build a user authentication system"

**Plan:**
```markdown
# Plan: Build a user authentication system

## Analysis
- Need login, registration, logout, password reset
- Should use secure password hashing
- May need JWT or session-based auth

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | Design database schema for users | medium |
| 2    | Implement password hashing | low |
| 3    | Create registration endpoint | medium |
| 4    | Create login endpoint | medium |
| 5    | Implement session/JWT auth | high |
| 6    | Add password reset flow | high |
| 7    | Write unit tests | medium |
| 8    | Security review | medium |

## Dependencies
- Step 1 → Step 2-7
- Step 7 before Step 8

## Considerations
- Use bcrypt for password hashing
- Implement rate limiting for auth endpoints
- Consider CSRF protection
```
