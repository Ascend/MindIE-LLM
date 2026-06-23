# PM-based Installation

The following uses MindIE LLM as an example to describe how to install MindIE on a physical machine.

1. To ensure secure file permissions after installation, run the following commands:

   ```bash
   old_umask=$(umask)
   umask 027
   ```

2. Run the following command to install the `.whl` package:

   ```bash
   pip install mindie_llm-{version}-{python_tag}-{platform_tag}.whl --no-deps
   ```

   > [!NOTE]NOTE
   >
   > - The preceding uses the `mindie_llm` package as an example. If you want to install MindIE Motor or MindIE SD, replace it with the corresponding `.whl` package name.
   > - If you need to use the source code for compilation and installation, go to the corresponding code repository to obtain the compilation guide. For example, for MindIE-LLM, see [build guide](../../../developer_guide/build_guide_llm.md).

3. If the following information is displayed, the software is successfully installed:

    ```text
    Successfully installed xxx
    ```

   *xxx* indicates the name of the actual software package to be installed.

4. (Optional) Run the following command to query the installation path:

   ```bash
   pip show mindie_llm | grep location
   ```

   If the Python version is 3.11, the default installation path is `/usr/local/lib/python3.11/site-packages`.
5. Run the following command to restore the permission:

   ```bash
   umask $old_umask
   ```
