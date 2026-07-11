# ─────────────────────────────────────────────────────────────────────────────
# Legal RAG — AWS Infrastructure
#
# Architecture:
#   ECR  →  ECS Fargate (backend container)
#             ↕ EFS (persistent ChromaDB index + raw PDFs)
#           ALB (public HTTPS endpoint)
#   Secrets Manager (ANTHROPIC_API_KEY)
#
# Services used:
#   - ECR:              Docker image registry
#   - ECS Fargate:      Serverless container runtime (no EC2 to manage)
#   - EFS:              Persistent file system for ChromaDB index (survives deploys)
#   - ALB:              Application Load Balancer, public entry point
#   - ACM:              TLS certificate (optional, see variables)
#   - Secrets Manager:  API key storage
#   - CloudWatch Logs:  Container log output
#   - VPC/Subnets/SGs:  Network isolation
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state — swap bucket/key for your own S3 bucket
  backend "s3" {
    bucket = "your-terraform-state-bucket"
    key    = "legal-rag/terraform.tfstate"
    region = "eu-west-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Data sources ──────────────────────────────────────────────────────────────

data "aws_availability_zones" "available" {}

# ── VPC ───────────────────────────────────────────────────────────────────────

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "${var.app_name}-vpc" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "${var.app_name}-public-${count.index}" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.app_name}-igw" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "${var.app_name}-rt" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ── Security Groups ───────────────────────────────────────────────────────────

resource "aws_security_group" "alb" {
  name   = "${var.app_name}-alb-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "${var.app_name}-alb-sg" }
}

resource "aws_security_group" "ecs" {
  name   = "${var.app_name}-ecs-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "${var.app_name}-ecs-sg" }
}

resource "aws_security_group" "efs" {
  name   = "${var.app_name}-efs-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }
  tags = { Name = "${var.app_name}-efs-sg" }
}

# ── ECR ───────────────────────────────────────────────────────────────────────

resource "aws_ecr_repository" "app" {
  name                 = var.app_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = { Name = var.app_name }
}

# Lifecycle: keep only the last 10 images to save storage costs
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ── EFS (persistent index storage) ───────────────────────────────────────────

resource "aws_efs_file_system" "index" {
  creation_token   = "${var.app_name}-index"
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"
  encrypted        = true
  tags             = { Name = "${var.app_name}-index" }
}

resource "aws_efs_mount_target" "index" {
  count           = 2
  file_system_id  = aws_efs_file_system.index.id
  subnet_id       = aws_subnet.public[count.index].id
  security_groups = [aws_security_group.efs.id]
}

# ── Secrets Manager ───────────────────────────────────────────────────────────

resource "aws_secretsmanager_secret" "anthropic_key" {
  name        = "${var.app_name}/anthropic-api-key"
  description = "Anthropic API key for LLM generation"
}

# Note: actual secret value set manually or via CI/CD, not in Terraform state
# aws secretsmanager put-secret-value --secret-id legal-rag/anthropic-api-key \
#   --secret-string '{"ANTHROPIC_API_KEY":"sk-ant-..."}'

# ── CloudWatch Log Group ──────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.app_name}"
  retention_in_days = 14
  tags              = { Name = var.app_name }
}

# ── IAM ───────────────────────────────────────────────────────────────────────

resource "aws_iam_role" "ecs_execution" {
  name = "${var.app_name}-ecs-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Allow ECS to read from Secrets Manager
resource "aws_iam_role_policy" "ecs_secrets" {
  name = "${var.app_name}-secrets-policy"
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = aws_secretsmanager_secret.anthropic_key.arn
    }]
  })
}

resource "aws_iam_role" "ecs_task" {
  name = "${var.app_name}-ecs-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────

resource "aws_ecs_cluster" "main" {
  name = var.app_name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = { Name = var.app_name }
}

# ── ECS Task Definition ───────────────────────────────────────────────────────

resource "aws_ecs_task_definition" "app" {
  family                   = var.app_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  # Mount EFS volume
  volume {
    name = "index-data"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.index.id
      root_directory     = "/"
      transit_encryption = "ENABLED"
    }
  }

  container_definitions = jsonencode([{
    name      = var.app_name
    image     = "${aws_ecr_repository.app.repository_url}:latest"
    essential = true

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "LLM_BACKEND",    value = "api" },
      { name = "API_MODEL",      value = "claude-haiku-4-5-20251001" },
      { name = "INDEX_DIR",      value = "/data/index" },
      { name = "DATA_RAW_DIR",   value = "/data/raw" },
      { name = "BM25_TOP_K",     value = "50" },
      { name = "DENSE_TOP_K",    value = "50" },
      { name = "RERANK_TOP_K",   value = "8" },
    ]

    secrets = [{
      name      = "ANTHROPIC_API_KEY"
      valueFrom = "${aws_secretsmanager_secret.anthropic_key.arn}:ANTHROPIC_API_KEY::"
    }]

    mountPoints = [{
      containerPath = "/data"
      sourceVolume  = "index-data"
      readOnly      = false
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.app.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "python3 -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\""]
      interval    = 30
      timeout     = 10
      retries     = 3
      startPeriod = 120
    }
  }])

  tags = { Name = var.app_name }
}

# ── ALB ───────────────────────────────────────────────────────────────────────

resource "aws_lb" "main" {
  name               = var.app_name
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
  tags               = { Name = var.app_name }
}

resource "aws_lb_target_group" "app" {
  name        = var.app_name
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }

  tags = { Name = var.app_name }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# ── ECS Service ───────────────────────────────────────────────────────────────

resource "aws_ecs_service" "app" {
  name            = var.app_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = var.app_name
    container_port   = 8000
  }

  # Allow rolling deploys without downtime
  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  depends_on = [aws_lb_listener.http]

  tags = { Name = var.app_name }
}
