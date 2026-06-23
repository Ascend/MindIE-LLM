# Enabling Live Restore

By default, when the Docker daemon terminates, it shuts down the running container. You can configure the daemon so that the container keeps running when the daemon is unavailable. This function is called live-restore.

It helps reduce container downtime due to daemon crashes, service interruptions, or upgrades.

If the Docker configuration is modified during task running, the Docker daemon needs to be reloaded. As a result, the Docker container restarts and services are interrupted, which poses risks. In this case, you can enable the live-restore function to keep the container active when the daemon is unavailable.

**Method 1**:

Add the configuration to the daemon configuration, that is, the `docker-daemon.json` file. The configuration is as follows:

```json
{
    "live-restore": true
}
```

**Method 2**:

Manually enable live-restore.

```bash
dockerd --live-restore systemd
```
