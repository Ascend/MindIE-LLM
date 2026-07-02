# Upgrade

## Installation Using the WHL Package

You can install the `.whl` package of the new version to upgrade MindIE. The following uses the upgrade of MindIE LLM as an example.

Run the following command to install the new `.whl` package to complete the upgrade:

```bash
pip install mindie_llm-{version}-{python_tag}-{platform_tag}.whl

```

> [!NOTE]NOTE
> The preceding uses the `mindie_llm` package as an example. If you want to upgrade MindIE Motor or MindIE SD, replace it with the corresponding `.whl` package name.

If the upgrade is performed between the same versions, add the `--force-reinstall` parameter to forcibly reinstall the package.

> [!CAUTION]NOTE
> During the reinstallation of MindIE LLM, the entire installation directory (`/mindie_llm`) will be deleted before installing the new version. If you need to retain configuration files, certificate files, etc., back them up in advance.

## Installation Using the RUN Package

To upgrade MindIE, obtain the software package of the target version and run the ```--upgrade``` command to overwrite the old version.

1. Log in to the installation environment as the installation user of the software package.
2. Go to the directory where the software package is stored.
3. Grant the execute permission on the software package.

    ```bash
    chmod +x software_package_name.run
    ```

4. Run the following command to check the consistency and integrity of the software package installation file:

    ```bash
    ./software_package_name.run --check
    ```

5. Upgrade the software.

   - If the installation path is specified during the installation, run the following command:

       ```bash
       ./software_package_name.run --upgrade --install-path=<path>
       ```

       In the preceding command, ```<path>``` indicates the specified installation directory of the software package. Replace the software package name with the actual one.

   - If the installation path is not specified during the installation, run the following command:

       ```bash
       ./software_package_name.run --upgrade
       ```

       > [!NOTE]NOTE
       > If the upgrade path is not specified, the software is upgraded to the default path. The default upgrade path is as follows:
       > - root user: /usr/local/Ascend
       > - Non-root user: /home/{current user name}/Ascend

       If you want to upgrade the software package on the premise that [Huawei Enterprise End User License Agreement (EULA)](https://e.huawei.com/en/about/eula) is signed by default, add ```--quiet``` to the upgrade command. For example, if you add this option to the end of ```./Software_package_name.run --upgrade```, step 6 will be skipped.

6. Sign [Huawei Enterprise End User License Agreement (EULA)](https://e.huawei.com/en/about/eula) to proceed to the installation process. Enter y or Y to confirm the agreement, and enter any other character to reject the agreement. After you accept the agreement, the upgrade starts.
If the current language environment does not meet the requirements, run the following command to configure the default language environment:

    ```bash
    # Set the language to Chinese (simplified).
    export LANG=zh_CN.UTF-8
    # Set the language to English.
    export LANG=en_US.UTF-8
    ```

7. After the upgrade, use either of the following methods to update dependencies.

   - Method 1: Install the missing dependencies and the dependencies whose versions are changed by referring to [Installing Dependencies.](installing_software_and_dependencies.md#installing-dependencies).
     For example, Numba is introduced to MindIE 2.0.RC1 compared with MindIE 1.0.0, and you can run the following command to install it.

   ```bash
   pip3 install numba==0.61.2
   ```

   - Method 2: Run the following command to install all dependencies required by the current version by referring to [Installing Dependencies.](installing_software_and_dependencies.md#installing-dependencies). (This method may overwrite the existing dependency versions in the environment.)

   ```bash
   pip3 install -r requirements.txt
   ```
