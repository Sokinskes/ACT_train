#!/bin/bash
# 构建并发布 Docker 镜像到 Docker Hub
# 用法: ./docker_build.sh [docker-hub-username] [tag]

set -e

DOCKER_HUB_USER="${1:-sokinskes}"
TAG="${2:-latest}"
IMAGE_NAME="act"

echo "🐳 开始构建 Docker 镜像..."
echo "镜像名: ${DOCKER_HUB_USER}/${IMAGE_NAME}:${TAG}"

# 构建开发版本
echo "📦 构建开发版本..."
docker build -t ${IMAGE_NAME}:dev -f Dockerfile .
docker tag ${IMAGE_NAME}:dev ${DOCKER_HUB_USER}/${IMAGE_NAME}:dev

# 构建运行时版本
echo "📦 构建运行时版本..."
docker build -t ${IMAGE_NAME}:runtime -f Dockerfile.runtime .
docker tag ${IMAGE_NAME}:runtime ${DOCKER_HUB_USER}/${IMAGE_NAME}:runtime

# 标记最新版本
docker tag ${IMAGE_NAME}:dev ${DOCKER_HUB_USER}/${IMAGE_NAME}:${TAG}

echo "✅ 构建完成！"
echo ""
echo "本地镜像列表:"
docker images | grep ${IMAGE_NAME}

echo ""
echo "📤 如需上传到 Docker Hub，运行:"
echo "  docker login"
echo "  docker push ${DOCKER_HUB_USER}/${IMAGE_NAME}:dev"
echo "  docker push ${DOCKER_HUB_USER}/${IMAGE_NAME}:runtime"
echo "  docker push ${DOCKER_HUB_USER}/${IMAGE_NAME}:${TAG}"
