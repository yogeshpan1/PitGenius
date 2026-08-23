# AWS Deployment (config written, NOT deployed — needs your credentials)

PitGenius is designed to deploy on AWS Academy-compatible services.

## Architecture

```
Route53 ──> CloudFront
              ├── S3 (React frontend build)
              └── ALB ── ECS Fargate (FastAPI container)
                            ├── EFS or S3 mount: data/pitgenius.db (read-mostly)
                            └── Secrets Manager: LLM_API_KEY
EventBridge (cron) ── Lambda: pre-race publish + post-race score steps
```

## Steps you must run yourself

1. **Build & push the API image**
   ```bash
   docker build -t pitgenius-api -f Dockerfile.api .
   # tag + push to ECR (create repo first):
   aws ecr create-repository --repository-name pitgenius/api
   docker tag pitgenius-api:latest <acct>.dkr.ecr.<region>.amazonaws.com/pitgenius/api:latest
   aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
   docker push <acct>.dkr.ecr.<region>.amazonaws.com/pitgenius/api:latest
   ```
2. **Frontend**: `cd frontend && npm run build`, upload `dist/` to S3,
   enable static hosting / CloudFront origin.
3. **Secrets**: store `LLM_API_KEY` in Secrets Manager; wire into the task
   definition.
4. **Scheduled jobs**: EventBridge rules calling the publish/score steps
   before/after each race weekend (see scripts/live_race.py).

Note: AWS Academy sandbox accounts cannot create custom domains or keep
resources between sessions — redeploy per session or use a personal account.