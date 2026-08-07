# 容器镜像构建

本目录下的文件用于构建 Triton Server 容器镜像。

运行以下命令构建 Triton Server 容器镜像：

```bash
docker build --file ./server.containerfile --tag <image_name_here> .
```

运行以下命令构建客户端负载生成容器镜像：

```bash
docker build --file ./client.containerfile --tag <image_name_here> .
```

> 💡 **AI Infra 视角**：这里的镜像构建做了"分容器"设计——server 镜像跑推理服务，client 镜像专门发请求做压测。生产实践中负载生成器（load generator）通常是独立部署的，不会和推理服务混在一起，这样既能独立扩缩压测客户端数量，也方便在 CI/CD 流水线中做发布前的性能回归验证。
