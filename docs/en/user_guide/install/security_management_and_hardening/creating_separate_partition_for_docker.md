# Creating a Separate Partition for Docker

After Docker is installed, the default directory `/var/lib/docker` stores Docker files, including images and containers. When the storage space of the directory is full, Docker and the host may be unavailable. For this reason, a partition (logical volume) needs to be created to save the Docker files.

- Create a separate partition on the newly installed device to mount the `/var/lib/docker` directory. For details, see [Table 1](#table1).

    **Table 1** Docker partitions <a id="table1"></a>

|Partition|Description|Size|Boot Flag|
|--|--|--|--|
|/boot|Boot partition.|500MB|on|
|/var|Partition for storing data generated during software running, such as logs and cache.|>300GB|off|
|/var/lib/docker|Partition for storing Docker images and containers.<br>Docker images and containers are stored in the `/var/lib/docker` partition by default. If the usage of the `/var/lib/docker` partition is greater than 85%, Kubernetes automatically triggers the resource eviction mechanism. Ensure that the usage of the `/var/lib/docker` partition is less than 85%.|>300GB|off|
|/|Primary partition|>300GB|off|

- For an installed system, use the Logical Volume Manager (LVM) to create a partition.
