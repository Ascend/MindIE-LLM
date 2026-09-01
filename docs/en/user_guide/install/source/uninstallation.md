# Uninstallation

## Installation Using the WHL Package

You can uninstall the `.whl` software package. The following uses MindIE LLM as an example.

Run the following command to uninstall the `.whl` package:

```bash
pip uninstall mindie_llm-{version}-{python_tag}-{platform_tag}.whl

```

> [!NOTE]
> The preceding uses the `mindie_llm` package as an example. To uninstall MindIE Motor CPP or MindIE SD, replace the package name with the corresponding `.whl` package name.

If the following information is displayed, the software is successfully uninstalled:

```bash

[INFO] XXX uninstall success

```

## Installation Using the RUN Package

To uninstall MindIE, you can run the uninstallation script or directly uninstall the software.

### Method 1: Script-based Uninstallation (Recommended)

You can run the uninstallation script to uninstall the software.

1. Go to the path where the uninstallation script is stored. Generally, the uninstallation script is stored in the scripts directory. (The path of the scripts directory is subject to the actual situation.)

    ```bash
    cd <path>/mindie/<version>/scripts
    ```

    `<path>` indicates the installation path of the software package. `<version>` indicates the software package version. Replace them as required.

2. Run the ```./uninstall.sh``` command to execute the script.
If the following information is printed, the software is successfully uninstalled:

```text
[INFO] xxx uninstall success
```

```xxx``` indicates the name of a software package to be uninstalled.

> [!NOTE]
> After the uninstallation is complete, you are advised to run the following command to cancel the configuration of the **TUNE_BANK_PATH** environment variable:
>
> ```bash
>unset TUNE_BANK_PATH
>```

### Method 2: Package-based Uninstallation

To uninstall an installed software package, perform the following steps:

1. Log in to the installation environment as the installation user of the software package.
2. Go to the directory where the software package is stored.
3. Run the following commands to uninstall the software package.

- If the installation path is specified during the installation, run the following command:
  
    ```bash
    ./software_package_name.run --uninstall --install-path=<path>
    ```

- If the installation path is not specified during the installation, run the following command:

    ```bash
    ./software_package_name.run --uninstall
    ```

If the following information is printed, the software is successfully uninstalled:

```text
[INFO] xxx uninstall success
```

```xxx``` indicates the name of the software package to be uninstalled.

> [!NOTE]
> After the uninstallation is complete, you are advised to run the following command to cancel the configuration of the TUNE_BANK_PATH environment variable:
>
> ```bash
> unset TUNE_BANK_PATH
> ```
