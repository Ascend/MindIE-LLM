# PM-based Installation

This document describes how to install MindIE on a physical machine. Select the `.whl` or `.run` package based on your requirements.

## Installation Using the WHL Package

1. To ensure secure file permissions after installation, run the following commands:

   ```bash
   old_umask=$(umask)
   umask 027
   ```

2. Run the following command to install the `.whl` package:

   ```bash
   pip install mindie_llm-{version}-{python_tag}-{platform_tag}.whl --no-deps
   ```

   > [!NOTE]
   >
   > - The preceding uses the `mindie_llm` package as an example. If you want to install MindIE Motor or MindIE SD, replace it with the corresponding `.whl` package name.
   > - If you need to use the source code for compilation and installation, go to the corresponding code repository to obtain the compilation guide. For example, for MindIE-LLM, see [build guide](../../../developer_guide/build_guide_llm.md).

3. If the following information is displayed, the software is successfully installed:

    ```text
    Successfully installed xxx
    ```

   *xxx* indicates the name of the actual software package to be installed.

4. (Optional) Run the following command to query the software version and installation path:

   ```bash
   pip show mindie_llm
   ```

   If the Python version is 3.11, the default installation path is `/usr/local/lib/python3.11/site-packages`.
5. Run the following command to restore the permission:

   ```bash
   umask $old_umask
   ```

## Installation Using the RUN Package

MindIE Motor, MindIE LLM, and MindIE SD will be installed in sequence when you install MindIE. The component packages are stored in the sub-path of MindIE.

1. Log in to the installation environment as the installation user of the CANN package.
2. Upload the obtained MindIE package to any path (for example, /home/package) in the installation environment.
3. Go to the directory where the software package is stored.

   ```bash
   cd /home/package
   ```

4. Grant the execute permission on the software package.

   ```bash
   chmod +x software_package_name.run
   ```

   software_package_name.run indicates the development toolkit Ascend-mindie_\<version>\_linux-\<arch>_\<abi>.run. Replace it with the actual package name.

5. Add environment variables of the Toolkit package. (The following uses the default installation path of the root user as an example.)

   ```bash
   source /usr/local/Ascend/cann/set_env.sh
   ```

6. Check the consistency and integrity of the software package installation file.

   ```bash
   ./software_package_name.run --check
   ```

7. Install the software. (The following command supports parameters such as --install-path=\<path>. For details about the parameters, see [Software Package Options](../faq_and_appendixes/software_package_options.md).)

   ```bash
   ./software_package_name.run --install --quiet
   ```

   > [!NOTE]
   >
   > - If the installation is performed by the root user, do not specify the installation path in the directory of a non-root user.
   > - If you do not specify an installation path, the software is installed in the default path. The default installation paths are as follows:
   >   - root user: /usr/local/Ascend
   >   - Non-root user: /home/{current user name}/Ascend
   > - The paths of software package installation logs are as follows:
   >   - root user: /var/log/mindie_log/mindie_install.log
   >   - Non-root user: /home/{current user name}/var/log/mindie_log/mindie_install.log
   > - The aie_tmp_source folder is temporarily generated in the current directory during the installation. After the installation is complete, the folder is deleted. If a folder with the same name already exists, it will be deleted after the installation.

   Once you execute the preceding commands, you agree to the terms and conditions of [Huawei Enterprise End User License Agreement (EULA)](https://e.huawei.com/en/about/eula).

   If the following information is printed, the software is successfully installed:

   ```text
   xxx install success
   ```

   ```xxx``` indicates the name of the software package to be installed.

8. (Optional) Run the following command to query the software version: The default installation path is used as an example:

   ```bash
   cat /usr/local/Ascend/mindie/latest/version.info
   ```

9. Configure environment variable.

A process-level environment variable setting script is provided to automatically set environment variables. The specified environment variables automatically become invalid after the user process ends. Example:

Configure environment variables in the default installation path of the root user:

```bash
source /usr/local/Ascend/mindie/set_env.sh
```

Configure environment variables in the default installation path of a non-root user:

```bash
source /home/{current_user_name}/Ascend/mindie/set_env.sh
```

You can also configure permanent environment variables by modifying the ```~/.bashrc``` file. The procedure is as follows:

   1. Run the ```vi ~/.bashrc``` command in any directory as the running user to open the ```.bashrc``` file and append the preceding lines to the file.
   2. Run the ```:wq!``` command to save the file and exit.
   3. Run the ```source ~/.bashrc``` command for the modification to take effect immediately.
