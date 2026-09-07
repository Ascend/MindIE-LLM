# Hardening Kubernetes Security

To ensure secure running of the environment, you are advised to control the login permission of the master node in a cluster based on services, and control the access permission of the Kubernetes private key file and authentication credentials stored in etcd. You are not advised to directly operate a Kubernetes cluster in the background.

Kubernetes needs to be hardened as follows:

- kube-controller

    Add `-serviceaccount` to `--controllers` in the YAML configuration file of kube-controller, to disable the default service account of the namespace. This prevents unnecessary service accounts from being generated in the user-defined namespace when cluster services are deployed.

- kube-proxy
  - Add `--nodeport-addresses` to the startup parameter of kube-proxy.
  - For the installed Kubernetes system, modify the ConfigMap of kube-proxy.

        ```linux
        kubectl edit cm kube-proxy -n kube-system
        ```

  - Manually change the value of `nodePortAddresses` in the ConfigMap to the node IP address in CIDR format.
  - Manually change the value of `healthzBindAddress` in the ConfigMap to the node IP address in CIDR format.
  - Apply the preceding configuration to the Kubernetes proxy. You can directly delete all pod tasks whose names contain "kube-proxy" in Kubernetes. Then, Kubernetes will directly restart the proxy service.

        ```linux
        kubectl delete pod {kube-proxy_pod_name} -n kube-system
        ```

- kube-apiserver
  - Add `--kubelet-certificate-authority` to configure the path of the kubelet CA certificate, which is used to verify the validity of the kubelet server certificate.
  - Change the value of `--profiling` to `false` to prevent users from dynamically changing the kube-apiserver log level.
  - Modify or add `--tls-cipher-suites` as follows to avoid risks caused by insecure TLS cipher suites.

      ```text
      --tls-cipher-suites=TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
      ```

  - Modify or add `--tls-min-version`. For example, you can set `--tls-min-version=VersionTLS13` to use TLS 1.3 for communication encryption during apiserver configuration.
  - Modify or add `--audit-policy-file` to configure the audit policy of Kubernetes. For details, see the official Kubernetes documentation.

- kubelet
  - To prevent a single pod from occupying too many processes, you can enable `SupportPodPidsLimit` and set `--pod-max-pids`. Add `--feature-gates=SupportPodPidsLimit=true --pod-max-pids=<max pid number>` to `KUBELET_KUBEADM_ARGS` in the kubelet configuration file. After the modification, restart the kubelet service for the modification to take effect. For details, see the official Kubernetes documentation.
  - Set `--address` or change the value of the `address` field in the startup configuration file to the host IP address.
  - Configure `--tls-min-version` or modify `tlsMinVersion` in the startup configuration file. For example, `tlsMinVersion: VersionTLS13` indicates that TLS 1.3 is used to encrypt communication during kubelet configuration.
  - Modify or add `--tls-cipher-suites` as follows to avoid risks caused by insecure TLS cipher suites.

      ```text
      --tls-cipher-suites=TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
      ```

      > [!NOTE]
      > Kubernetes v1.19 and later versions support TLS 1.3 cipher suites. It is recommended that TLS 1.3 cipher suites be added when Kubernetes of a later version is used.

- If the OS kernel version used by the Kubernetes cluster is 4.6 or later, manually enable AppArmor or SELinux after Kubernetes is installed.
- To make the bandwidth limit of the inference service pod take effect, install the bandwidth plugin in the CNI bin directory (`/opt/cni/bin` by default), modify the CNI configuration file (`/etc/cni/net.d` by default), and add `bandwidth` to `plugins`.

  ```json
    {
    "type": "bandwidth",
    "capabilities": {
      "bandwidth": true
      }
    }
  ```

- Workload security:
  - Do not use privileged containers to start pods.
  - Do not allow pod containers to share the IPC, network, and process ID namespaces of the host.
  - You are advised not to run pod containers as the `root` user.
  - Minimize the capabilities required by pod containers.
  - Ensure that the maximum CPU and memory usage is set for pods.
  - Ensure that no container is mounted with Docker Socket.
  - It is recommended that the `securityContext` of pods use a read-only file system.
  - Ensure that `allowPrivilegeEscalation` is set to `false` in `securityContext` of pods.

- For details about other security hardening items, see "Security" at the Kubernetes official website or in other vendors' security hardening solutions.
