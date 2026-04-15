# Kubernetes安全加固

为保证环境安全运行，建议用户根据业务控制集群Master节点登录权限，对环境中Kubernetes的私钥文件以及etcd中存储的认证凭据做好访问权限控制；不建议用户在后台直接操作Kubernetes集群。

Kubernetes需要进行如下加固：

- kube-controller加强：

    在kube-controller的yaml配置文件中启动参数“--controllers“内添加子项“-serviceaccount-token“，用于禁用命名空间默认服务账号。防止用户进行集群服务部署时，在用户定义命名空间内产生不需要使用的服务账号。

- kube-proxy加强：
  - 在“kube-proxy“启动参数中加入“--nodeport-addresses“参数。
  - 针对已经安装的Kubernetes系统，通过如下命令修改kube-proxy的configmap。

        ```linux
        kubectl edit cm kube-proxy -n kube-system
        ```

  - 手动修改configmap中的“nodePortAddresses“参数为CIDR格式的节点IP。
  - 手动修改configmap中的“healthzBindAddress“参数为CIDR格式的节点IP。
  - 将上述配置应用到k8s proxy，用户可直接删除K8s中名字带kube-proxy的所有Pod任务，后续K8s会直接重新拉起proxy服务。

        ```linux
        kubectl delete pod {kube-proxy pod名称} -n kube-system
        ```

- kube-apiserver加强：
  - 添加启动参数“--kubelet-certificate-authority“，配置kubelet CA证书路径，用于验证kubelet服务端证书有效性。
  - 修改启动参数“--profiling“的值设置为“false“，防止用户动态修改kube-apiserver日志级别。
  - 修改或增加启动参数“--tls-cipher-suites“，避免使用不安全的TLS加密套件带来风险，设置如下所示：

        ```text
        --tls-cipher-suites=TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_RSA_WITH_AES_128_GCM
        _SHA256,TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,TLS_ECDHE_RSA_WITH_AES_256_GC
        M_SHA384,TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,TLS_ECDHE_ECDSA_WITH_AES_256_G
        CM_SHA384
        ```

  - 修改或增加启动参数“--tls-min-version“，取值示例：--tls-min-version=VersionTLS13，用于配置apiserver时使用TLS1.3安全协议对通信进行加密。
  - 修改或增加启动参数“--audit-policy-file“，配置Kubernetes的审计策略，具体配置可参考Kubernetes官方文档。

- kubelet加强：
  - 为防止单Pod占用过多进程数，可以开启“SupportPodPidsLimit“，并设置“--pod-max-pids“参数。在kubelet配置文件的“KUBELET\_KUBEADM\_ARGS“项增加--feature-gates=SupportPodPidsLimit=true --pod-max-pids=\<max pid number\>。配置修改后，重启生效。详细配置信息可参考Kubernetes官方文档。
  - 配置启动参数“--address“或者修改启动配置文件中的address字段，设置值为主机IP。
  - 配置启动参数“--tls-min-version“或者修改启动配置文件中的tlsMinVersion字段，启动配置文件字段取值示例：tlsMinVersion:VersionTLS13，用于配置kubelet时使用TLS1.3安全协议对通信进行加密。
  - 修改或增加启动参数“--tls-cipher-suites“，避免使用不安全的TLS加密套件带来风险，设置如下所示：

        ```text
        --tls-cipher-suites=TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_RSA_WITH_AES_128_GCM
        _SHA256,TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,TLS_ECDHE_RSA_WITH_AES_256_GC
        M_SHA384,TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,TLS_ECDHE_ECDSA_WITH_AES_256_G
        CM_SHA384
        ```

        > [!NOTE]说明
        > Kubernetes v1.19及以上版本支持TLSv1.3的加密套件，建议使用高版本的Kubernetes时，加上TLSv1.3的加密套件。

- 若Kubernetes集群使用的OS kernel内核版本大于等于4.6，安装完Kubernetes后手动开启AppArmor或者SELinux。
- 为使推理服务Pod的带宽限制生效，需要安装bandwidth插件到CNI bin目录中（默认为/opt/cni/bin），并修改CNI配置文件（默认路径：/etc/cni/net.d），在plugins中加入bandwidth。

     ```json

     {
     "type": "bandwidth",
     "capabilities": {"bandwidth": true}
     }

     ```

- 工作负载安全：
  - 禁止使用特权容器启动Pod。
  - 禁止Pod容器共享主机的IPC、网络、进程ID命名空间。
  - 建议不要以root权限运行Pod容器。
  - 最小化配置Pod容器需要的capabilities权限。
  - 确保Pod设置CPU和内存的最大申请限制。
  - 确保不使用挂载了Docker Socket的容器。
  - 建议Pod的securityContext使用只读文件系统。
  - 确保Pod的securityContext限制allowPrivilegeEscalation为false。

- 其余安全加固内容可参考Kubernetes官方文档Security相关内容，也可以参考业界其他优秀加固方案。
