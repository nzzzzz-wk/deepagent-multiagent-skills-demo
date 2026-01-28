---
name: devops
description: Planning skills for DevOps and infrastructure - CI/CD, containers, deployment, monitoring
---

# DevOps Planning Skill

## When Triggered
This skill activates when the user asks about:
- CI/CD pipeline setup and configuration
- Docker containerization
- Kubernetes deployment
- Infrastructure as Code (Terraform, CloudFormation)
- Monitoring and logging setup
- Cloud deployment (AWS, GCP, Azure)
- Release management and rollback strategies

## Planning Principles
1. Define deployment strategy first (blue-green, canary, rolling)
2. Plan for environment parity (dev/staging/prod)
3. Include rollback procedures and disaster recovery
4. Plan for monitoring, alerting, and observability
5. Consider security scanning and compliance
6. Implement infrastructure as code
7. Plan for secret management

## DevOps Plan Template

```markdown
# Plan: {task_name}

## Infrastructure Overview
- Environment: {dev/staging/prod}
- Cloud Provider: {AWS/GCP/Azure}
- Resources: {compute, storage, network}

## Deployment Strategy
- Type: {rolling/blue-green/canary}
- Rollback: {how_to_rollback}
- Downtime: {expected_downtime}

## CI/CD Pipeline
- Trigger: {git_events}
- Stages: {build, test, deploy}
- Tests Required: {test_types}

## Monitoring & Observability
- Metrics: {what_to_monitor}
- Logging: {log_aggregation}
- Alerts: {alert_thresholds}

## Security
- Secret Management: {vault/env/secrets_manager}
- Scanning: {security_scan_tools}
- Compliance: {compliance_requirements}

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | {step_1}    | low        |
| 2    | {step_2}    | medium     |

## Dependencies
{dependencies}

## Considerations
{cost, scaling, etc.}
```

## Examples

### Example 1: CI/CD Pipeline
**Request:** "Set up CI/CD pipeline for a Node.js app"

**Plan:**
```markdown
# Plan: Set up CI/CD pipeline for Node.js app

## Infrastructure Overview
- Environment: GitHub Actions → AWS
- Cloud Provider: AWS (ECR + ECS)
- Resources: Containerized application

## Deployment Strategy
- Type: Blue-green deployment
- Rollback: Previous task definition
- Downtime: Zero (blue-green switch)

## CI/CD Pipeline
- Trigger: On push to main, PR to main
- Stages:
  1. Lint and format check
  2. Unit tests (Jest)
  3. Integration tests
  4. Build Docker image
  5. Security scan (Trivy)
  6. Push to ECR
  7. Deploy to ECS
- Tests Required: Unit, Integration

## Monitoring & Observability
- Metrics: CPU, memory, request latency, error rate (CloudWatch)
- Logging: CloudWatch Logs
- Alerts: Error rate > 1%, Latency > 500ms

## Security
- Secret Management: GitHub Secrets
- Scanning: Trivy for vulnerabilities
- Compliance: None specific

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | Create Dockerfile and docker-compose | low |
| 2    | Write GitHub Actions workflow | medium |
| 3    | Set up ECR repository | low |
| 4    | Configure ECS cluster | medium |
| 5    | Create CI pipeline (build + test) | medium |
| 6    | Create CD pipeline (deploy) | high |
| 7    | Set up monitoring dashboards | medium |
| 8    | Test deployment and rollback | high |

## Dependencies
- Step 1 → Step 5
- Step 5 → Step 6
- Step 6 → Step 7-8

## Considerations
- Keep Docker image size small
- Use caching for faster builds
- Set up cost alerts
```

### Example 2: Kubernetes Deployment
**Request:** "Deploy our app to Kubernetes"

**Plan:**
```markdown
# Plan: Deploy app to Kubernetes

## Infrastructure Overview
- Environment: AWS EKS
- Cloud Provider: AWS
- Resources: EKS cluster, RDS, ElastiCache

## Deployment Strategy
- Type: Rolling deployment with readiness checks
- Rollback: kubectl rollout undo
- Downtime: None (rolling update)

## Kubernetes Resources
- Deployment (app replicas)
- Service (internal/external load balancer)
- Ingress (with ALB)
- ConfigMap (config)
- Secret (sensitive data)
- HPA (autoscaling)

## CI/CD Pipeline
- Trigger: Tag push
- Stages: Build → Scan → Push to registry → kubectl apply
- Tests Required: Integration tests against staging

## Monitoring & Observability
- Metrics: Prometheus + Grafana
- Logging: Fluent Bit → CloudWatch
- Alerts:PagerDuty for critical

## Security
- Secret Management: External Secrets Operator
- Scanning: Trivy in CI, Falco at runtime
- Compliance: Pod security policies

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | Create Kubernetes manifests | medium |
| 2    | Set up EKS cluster | high |
| 3    | Configure ingress and TLS | medium |
| 4    | Set up secrets management | medium |
| 5    | Configure HPA | medium |
| 6    | Set up monitoring (Prometheus) | high |
| 7    | Configure logging pipeline | medium |
| 8    | Create deployment pipeline | high |
| 9    | Disaster recovery test | high |

## Dependencies
- Step 2 → Step 3-9
- Step 3 → Step 4-5
- Step 6-7 → Step 8

## Considerations
- Use namespaces for environments
- Set resource limits
- Plan for cluster autoscaling
```
