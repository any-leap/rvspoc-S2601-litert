# Gotchas

> 踩过的坑 + 恢复步骤。每条独立编号 `GOT-NNN`，只增不减。

## GOT-001 [网络/build] 大型依赖的 git clone 在 docker 容器内挂掉

### 症状
LiteRT CMake 配置阶段触发 `FetchContent` 拉 TensorFlow 仓库（v2.21.0-rc0，shallow clone 仍有几百 MB），git clone 中途断掉：

```
error: 5399 bytes of body are still expected
fetch-pack: unexpected disconnect while reading sideband packet
fatal: 过早的文件结束符（EOF）
fatal: fetch-pack：无效的 index-pack 输出
CMake Error: tensorflow-populate-gitclone.cmake:39 (message)
ninja: build stopped: subcommand failed.
```

### 原因
两个独立因素叠加：
1. TF 仓库即便 shallow 也太大（几十万 objects，git pack 走 sideband 协议对长连接稳定性敏感）
2. 即使 macOS host 配了 TUN 全局代理，**docker 容器在某些场景下未必走得到**（取决于 docker 网络模式 + TUN 实现），裸连 GitHub 时容易被限速/掐断

### 恢复步骤
绕开 FetchContent 在容器内 clone TF：用 host curl 直接拉 tarball 解压到 `.cache/tensorflow-src/`，build 脚本检测到就用 `-DTENSORFLOW_SOURCE_DIR=` 指过去。

```bash
# 在 host (macOS) 执行
mkdir -p .cache && cd .cache
curl -L --retry 5 --retry-all-errors -C - -o tf.tar.gz \
  https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.21.0-rc0.tar.gz
tar xzf tf.tar.gz && mv tensorflow-2.21.0-rc0 tensorflow-src && rm tf.tar.gz
```

`scripts/build_host.sh` 已自动检测 `.cache/tensorflow-src/` 存在则使用。

如果其它小依赖（abseil/eigen/ruy 等，每个 < 10MB）也挂，下一步是给 docker 透传 host 代理：在 `run.sh` 加 `-e HTTP_PROXY=http://host.docker.internal:<port>` 等环境变量。

#docker #build #network #gotcha

## GOT-002 [build/oom] LiteRT 编译时 cc1plus 被 OOM killer 干掉

### 症状
host build 跑到 LiteRT 自身 .cc 文件编译阶段，多个并行 cc1plus 进程被 SIGKILL：

```
FAILED: CMakeFiles/tensorflow-lite.dir/kernels/concatenation.cc.o
c++: fatal error: Killed signal terminated program cc1plus
FAILED: CMakeFiles/tensorflow-lite.dir/kernels/conv.cc.o
c++: fatal error: Killed signal terminated program cc1plus
ninja: build stopped: subcommand failed.
```

### 原因
LiteRT 的 `tflite/kernels/*.cc`（concatenation、conv、pooling 等）大量展开 Eigen / ruy / gemmlowp 模板，单个 cc1plus 编译占用 **2–4 GB RAM**。Docker Desktop on macOS 默认 VM 内存 ~8 GB，nproc=14，`-j$(nproc)` 起 14 个并行 cc1plus，必然 OOM。

### 恢复步骤
- 短期：build 脚本默认 `-j4`，激进的可以 `BUILD_JOBS=2`
- 中期：调高 Docker Desktop 的 Resources → Memory（建议 ≥ 16 GB），然后 `BUILD_JOBS=8`
- 验证：`docker run --rm rvspoc-s2601 free -h` 看容器内可用内存

```bash
# 重跑 build（脚本已 default -j4）
docker run -d --name rvspoc-host-build -v "$(pwd):/work" -w /work \
  rvspoc-s2601:latest bash scripts/build_host.sh
```

#docker #build #oom #gotcha
