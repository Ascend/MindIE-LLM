# Software Package Options

One-click software installation is supported in the CLI. You can select commands as required to complete the installation. All options below are optional.

Install command format: `./<package-name>.run [options]`

For details, see [Table 1](#table1).

**Table 1** Software package options <a id="table1"></a>

|Option|Description|
|--|--|
|--help \| -h|Queries help information.|
|--info|Queries software package build information.|
|--list|Queries the software package list.|
|--check|Checks the consistency and integrity of software packages.|
|--quiet \| -q|Silent installation, which skips interactive messages.|
|--extract=\<path\>|Decompresses the software package to a specified directory.|
|--install-path=\<path\>|This option can be set only to an absolute path.<br>Check the file in the following directories: <ul><li>For root user: `/etc/Ascend` </li><li>For non-root user: `/home/{username}/Ascend` </li></ul>If not specified, the installation defaults to: <ul><li>For root user: `/usr/local/Ascend` - </li><li>For non-root user: `/home/{username}/Ascend` </li></ul>If this option is used to specify the installation directory, the running user must have the read and write permissions on the specified installation directory.|
|--install|Installs a software package. You can specify the installation path `--install-path=<path>` or use the default installation path.|
|--upgrade|Upgrades the installed software from an earlier version to a later version. You can specify the upgrade path `--install-path=<path>` or use the default upgrade path.<br>After this option is used for an upgrade, the new version inherits the configuration in the `config.json` file of Server of an earlier version.|
|--uninstall|Uninstalls the software that has been installed. You can specify the uninstallation path `--install-path=<path>`.|
