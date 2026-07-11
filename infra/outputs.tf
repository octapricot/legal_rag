output "alb_dns_name" {
  description = "Public URL of the application (ALB DNS name)"
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "ECR repository URL — used in CI/CD to push images"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name — used in CI/CD to trigger deployments"
  value       = aws_ecs_cluster.main.name
}

output "efs_file_system_id" {
  description = "EFS file system ID — mount to pre-populate the index"
  value       = aws_efs_file_system.index.id
}
