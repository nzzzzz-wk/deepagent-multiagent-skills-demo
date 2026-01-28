---
name: web-dev
description: Planning skills for web development tasks - frontend, backend, APIs, databases
---

# Web Development Planning Skill

## When Triggered
This skill activates when the user asks about:
- Building web applications or websites
- Creating REST APIs or GraphQL endpoints
- Frontend development (React, Vue, Angular, etc.)
- Backend development (Node.js, Python, Go, etc.)
- Database design for web applications
- Authentication and authorization systems

## Planning Principles
1. Design API contracts first before implementation
2. Consider frontend/backend separation of concerns
3. Plan for authentication/authorization early
4. Include database schema design
5. Consider performance and caching strategies
6. Plan for error handling and validation
7. Consider scalability from the start

## Web Development Plan Template

```markdown
# Plan: {task_name}

## Architecture
- Frontend: {frontend_stack}
- Backend: {backend_stack}
- Database: {database}
- API Style: {REST/GraphQL}

## Database Design
{schema_overview}

## API Design
{endpoints_summary}

## Frontend Structure
{component_architecture}

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | {step_1}    | low        |
| 2    | {step_2}    | medium     |

## Dependencies
{dependencies}

## Considerations
{additional_considerations}
```

## Examples

### Example 1: REST API
**Request:** "Create a REST API for a blog"

**Plan:**
```markdown
# Plan: Create a REST API for a blog

## Architecture
- Backend: Python Fastapi
- Database: Supabase
- API Style: REST

## Database Design
- Users table (id, email, password_hash, created_at)
- Posts table (id, title, content, author_id, created_at)
- Comments table (id, content, post_id, author_id, created_at)

## API Design
- GET /posts - List all posts
- GET /posts/:id - Get single post
- POST /posts - Create post (auth required)
- PUT /posts/:id - Update post (auth required)
- DELETE /posts/:id - Delete post (auth required)
- POST /posts/:id/comments - Add comment (auth required)

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | Set up project with Express | low |
| 2    | Design database schema | medium |
| 3    | Create User model and auth | high |
| 4    | Create Post CRUD endpoints | medium |
| 5    | Create Comment endpoints | medium |
| 6    | Add validation and error handling | medium |
| 7    | Write unit tests | medium |

## Dependencies
- Step 1 → Step 2
- Step 3 → Step 4-6

## Considerations
- Use JWT for authentication
- Implement pagination for lists
- Add rate limiting
```

### Example 2: Full Stack App
**Request:** "Build a task management app"

**Plan:**
```markdown
# Plan: Build a task management app

## Architecture
- Frontend: React + TailwindCSS
- Backend: Node.js + Express
- Database: MongoDB
- API Style: REST

## Database Design
- Users (id, email, name)
- Projects (id, name, owner_id)
- Tasks (id, title, description, status, project_id, assignee_id)

## API Design
- /auth/* - Authentication endpoints
- /users/* - User management
- /projects/* - Project CRUD
- /tasks/* - Task CRUD

## Frontend Structure
- Pages: Login, Register, Dashboard, ProjectView, TaskDetail
- Components: TaskCard, ProjectList, Navbar

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | Set up full stack project structure | low |
| 2    | Implement user auth (register/login) | high |
| 3    | Create database models | medium |
| 4    | Build API endpoints | medium |
| 5    | Create React frontend pages | medium |
| 6    | Connect frontend to API | medium |
| 7    | Add real-time updates (optional) | high |

## Dependencies
- Step 1 → Step 2-4
- Step 4 → Step 5-6

## Considerations
- Use context for auth state
- Implement optimistic UI updates
- Add loading states
```
