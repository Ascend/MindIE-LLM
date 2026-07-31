# Hardening Security for Docker Container Running

To ensure secure running of a Docker container, you are advised to configure the following hardening items based on service requirements. For details, see the official description.

- Enable the AppArmor capability. You can specify the AppArmor file when running the container to protect the Linux system and applications because AppArmor provides security policies. Before enabling the AppArmor capability, enable the AppArmor function of the Linux kernel.
- Enable the SELinux capability. You can specify the SELinux configuration when running the container to improve security. Before enabling this function, you need to use `--selinux-enabled` to make the configuration take effect in the Docker daemon.
- Set system resource quotas for the container to prevent the container from exhausting the resources. System resources include but not limited to CPU and memory.
- Do not run untrusted applications in the container.
- Do not listen to unnecessary ports in the container.
- Configure a proper CPU priority for the container.
- Mount the root file system of the container in read-only mode.
- Bind the imported container traffic to a specific host interface and configure a specified IP address for the port mapping of the container.
- Limit the number of file handles and fork processes used for container running.
- Enable the authentication and encrypted transmission mechanisms for service ports for external listening of container services to prevent service data from being stolen.
- Do not run the SSH server in the container.
- Do not share namespaces, including the network namespace, UTS namespace, and user namespace.
- Do not mount **docker.sock** to the container.
- Ensure that no user is added to the Docker user group.
- Exercise caution when configuring parameters such as environment variables and configuration files during API call related to container or template creation or update, and ensure that secure images are used. Do not transfer sensitive information through environment variables or ConfigMaps to prevent sensitive data leakage or privilege escalation risks caused by improper configuration. You are advised to fully verify data before using it based on your services.
- Create a non-root user in the base image, start the image and process as the non-root user, and grant only necessary capabilities to the user to prevent security risks such as container escape caused by high-privileged users.
- Properly control the owners and permissions of files in the image to prevent security risks such as container escape caused by unnecessarily unauthorized access.
- Fix vulnerabilities in the base image in a timely manner.
- When distributing images, you are advised to enable the content trust function of Docker.
- Use the latest Docker version. You are advised to update the Docker version in a timely manner to prevent known vulnerabilities. Also, you are advised to perform security hardening on the host where the Docker container is running and periodically scan vulnerabilities.
