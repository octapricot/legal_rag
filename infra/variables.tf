variable "app_name" {
  description = "Application name — used as prefix for all AWS resources"
  type        = string
  default     = "legal-rag"
}

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "eu-west-1"
}

variable "task_cpu" {
  description = "ECS task CPU units (1024 = 1 vCPU). Min 2048 for embedding model."
  type        = number
  default     = 2048
}

variable "task_memory" {
  description = "ECS task memory in MB. Min 4096 for embedding model + ChromaDB."
  type        = number
  default     = 8192
}

variable "desired_count" {
  description = "Number of ECS task replicas"
  type        = number
  default     = 1
}
