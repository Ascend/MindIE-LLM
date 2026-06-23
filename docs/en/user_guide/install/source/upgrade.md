# Upgrade

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
