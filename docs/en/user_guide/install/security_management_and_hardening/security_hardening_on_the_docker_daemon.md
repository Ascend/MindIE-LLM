# Hardening Security on the Docker Daemon

- Restrict network access between containers. By default, the Docker daemon allows containers to communicate with each other, which may cause information leakage. You are advised to add `-icc=false` to the Docker daemon.
- Disable the remote management interface of the daemon. Do not use the Docker Remote API service. Strictly control the read and write permissions on the `docker.sock` file and add only necessary users to the Docker user group. If the Docker Remote API service needs to be enabled, you are advised to use `--authorization-plugin` to enable fine-grained access control for the daemon.
- Limit the numbers of file handles and fork processes in the container. You are advised to add the `nofile` and `nproc` parameters in `--default-ulimit` to the Docker daemon to prevent the host from being attacked by fork bombs or file handle exhaustion. The limits on container resources must be evaluated based on services. If the limits are improper, the container cannot run. For example, `--default-ulimit nofile=64:64 --default-ulimit nproc=512:512` indicates that the number of file handles of a single process is limited to 64, and the number of fork processes of a single UID user is limited to 512.
- Disable the userland proxy. You are advised to add `--userland-proxy=false` to the Docker daemon to reduce the attack surface.
- Enable the user namespace. After it is enabled, permissions of container users and host users are isolated. For example, enable this function in the Docker daemon using `--userns-remap=default`.
- Do not use the AUFS storage driver because it is not supported.
- Configure the log driver. Determine whether to enable the log driver based on service requirements.
