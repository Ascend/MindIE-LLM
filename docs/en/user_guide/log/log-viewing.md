# Log Viewing

By default,  MindIE  collects logs of the Informational level or higher.[Table 1](#en-us_topic_0000002140349885_table61551496213)  describes the default log flush paths. For details about how to set the flush path, see  [Setting the Log Flush Path](./setting_log_path.md).

**Table  1**  Log paths

<a name="en-us_topic_0000002140349885_table61551496213"></a>
<table><thead align="left"><tr id="en-us_topic_0000002140349885_row51551691528"><th class="cellrowborder" valign="top" width="36.76%" id="mcps1.2.3.1.1"><p id="en-us_topic_0000002140349885_p91551993214"><a name="en-us_topic_0000002140349885_p91551993214"></a><a name="en-us_topic_0000002140349885_p91551993214"></a>Path</p>
</th>
<th class="cellrowborder" valign="top" width="63.239999999999995%" id="mcps1.2.3.1.2"><p id="en-us_topic_0000002140349885_p51551491324"><a name="en-us_topic_0000002140349885_p51551491324"></a><a name="en-us_topic_0000002140349885_p51551491324"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="en-us_topic_0000002140349885_row915514912218"><td class="cellrowborder" valign="top" width="36.76%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000002140349885_p91551393210"><a name="en-us_topic_0000002140349885_p91551393210"></a><a name="en-us_topic_0000002140349885_p91551393210"></a>~/mindie/log</p>
</td>
<td class="cellrowborder" valign="top" width="63.239999999999995%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000002140349885_p14155139920"><a name="en-us_topic_0000002140349885_p14155139920"></a><a name="en-us_topic_0000002140349885_p14155139920"></a>Default log flush path</p>
</td>
</tr>
<tr id="en-us_topic_0000002140349885_row21551593211"><td class="cellrowborder" valign="top" width="36.76%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000002140349885_p11551295216"><a name="en-us_topic_0000002140349885_p11551295216"></a><a name="en-us_topic_0000002140349885_p11551295216"></a>~/mindie/log/security</p>
</td>
<td class="cellrowborder" valign="top" width="63.239999999999995%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000002140349885_p131551598215"><a name="en-us_topic_0000002140349885_p131551598215"></a><a name="en-us_topic_0000002140349885_p131551598215"></a>Security log path that is automatically generated in the default log flush path</p>
</td>
</tr>
<tr id="en-us_topic_0000002140349885_row915513914212"><td class="cellrowborder" valign="top" width="36.76%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000002140349885_p1155139223"><a name="en-us_topic_0000002140349885_p1155139223"></a><a name="en-us_topic_0000002140349885_p1155139223"></a>~/mindie/log/debug</p>
</td>
<td class="cellrowborder" valign="top" width="63.239999999999995%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000002140349885_p7155189626"><a name="en-us_topic_0000002140349885_p7155189626"></a><a name="en-us_topic_0000002140349885_p7155189626"></a>Run/Debug log path that is automatically generated in the default log flush path</p>
</td>
</tr>
</tbody>
</table>

The log file name format is mindie-_Component name_\_pid\_datetime.log. You can locate a log file based on the component name, process ID, and timestamp.

[**Example 1**]  MindIE Motor log file:

```bash
mindie-service_123_202410080206.log
```

You can run the  **cat** _Log file_  command to view logs.

[**Example 2**] View the  MindIE Motor  log file:

```bash
cat mindie-service_123_202410080206.log
```
