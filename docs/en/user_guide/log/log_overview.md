# Log Overview

## Log Type

Currently,  MindIE  logs are classified into security audit logs and run/debug logs. Any information that is related to login authentication, account management, access control, network attacks, and other security events during system running must be recorded in security audit logs. Except security audit logs, logs generated during service and debugging are run/debug logs.

## Log Format

Log format of all  MindIE  components:

```text
[date time] [pid] [tid] [Component name] [Log level in uppercase] [file:line] : [error code] [*] log message
```

>[!NOTE]
>The asterisk * indicates that if a component contains subcomponents or smaller functional modules, they are displayed prior to the log information.

The content in bold must be filled, and other fields are optional. You can configure the fields using the environment variable  **MINDIE\_LOG\_VERBOSE**. For details, see [Configuring the Log Content](configuring_log_content.md).

**Table  1**  Log field description

<a name="en-us_topic_0000002104910598_en-us_topic_0225421598_table970441113131"></a>
<table><thead align="left"><tr id="en-us_topic_0000002104910598_en-us_topic_0225421598_row12706191110135"><th class="cellrowborder" valign="top" width="32.019999999999996%" id="mcps1.2.3.1.1"><p id="en-us_topic_0000002104910598_en-us_topic_0225421598_p17706121171311"><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p17706121171311"></a><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p17706121171311"></a>Field</p>
</th>
<th class="cellrowborder" valign="top" width="67.97999999999999%" id="mcps1.2.3.1.2"><p id="en-us_topic_0000002104910598_en-us_topic_0225421598_p117061711201310"><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p117061711201310"></a><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p117061711201310"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="en-us_topic_0000002104910598_en-us_topic_0225421598_row248251311920"><td class="cellrowborder" valign="top" width="32.019999999999996%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000002104910598_en-us_topic_0225421598_p154833131596"><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p154833131596"></a><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p154833131596"></a><strong id="en-us_topic_0000002104910598_b115715511527"><a name="en-us_topic_0000002104910598_b115715511527"></a><a name="en-us_topic_0000002104910598_b115715511527"></a>date time</strong></p>
</td>
<td class="cellrowborder" valign="top" width="67.97999999999999%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000002104910598_en-us_topic_0225421598_p1948331319916"><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p1948331319916"></a><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p1948331319916"></a>Date and time.</p>
</td>
</tr>
<tr id="en-us_topic_0000002104910598_en-us_topic_0225421598_row26961771014"><td class="cellrowborder" valign="top" width="32.019999999999996%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000002104910598_en-us_topic_0225421598_p59912054121019"><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p59912054121019"></a><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p59912054121019"></a>pid</p>
</td>
<td class="cellrowborder" valign="top" width="67.97999999999999%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000002104910598_p33681119125311"><a name="en-us_topic_0000002104910598_p33681119125311"></a><a name="en-us_topic_0000002104910598_p33681119125311"></a>Process ID.</p>
</td>
</tr>
<tr id="en-us_topic_0000002104910598_en-us_topic_0225421598_row119915547102"><td class="cellrowborder" valign="top" width="32.019999999999996%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000002104910598_en-us_topic_0225421598_p194791732114"><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p194791732114"></a><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p194791732114"></a>tid</p>
</td>
<td class="cellrowborder" valign="top" width="67.97999999999999%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000002104910598_p17368131913534"><a name="en-us_topic_0000002104910598_p17368131913534"></a><a name="en-us_topic_0000002104910598_p17368131913534"></a>Thread ID.</p>
</td>
</tr>
<tr id="en-us_topic_0000002104910598_row2469112544119"><td class="cellrowborder" valign="top" width="32.019999999999996%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000002104910598_p1470925194112"><a name="en-us_topic_0000002104910598_p1470925194112"></a><a name="en-us_topic_0000002104910598_p1470925194112"></a>Component name</p>
</td>
<td class="cellrowborder" valign="top" width="67.97999999999999%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000002104910598_p1047082584113"><a name="en-us_topic_0000002104910598_p1047082584113"></a><a name="en-us_topic_0000002104910598_p1047082584113"></a>Name of the MindIE component. The options are as follows: [motor, server, llm, llmmodels, sd].</p>
</td>
</tr>
<tr id="en-us_topic_0000002104910598_row0298193664310"><td class="cellrowborder" valign="top" width="32.019999999999996%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000002104910598_p929933674316"><a name="en-us_topic_0000002104910598_p929933674316"></a><a name="en-us_topic_0000002104910598_p929933674316"></a><strong id="en-us_topic_0000002104910598_b05217710419"><a name="en-us_topic_0000002104910598_b05217710419"></a><a name="en-us_topic_0000002104910598_b05217710419"></a>Log level in uppercase</strong></p>
</td>
<td class="cellrowborder" valign="top" width="67.97999999999999%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000002104910598_p6299163684315"><a name="en-us_topic_0000002104910598_p6299163684315"></a><a name="en-us_topic_0000002104910598_p6299163684315"></a>Log level in uppercase. For details about the log levels, see <a href="setting_log_level.md#table1">Table 1</a>.</p>
</td>
</tr>
<tr id="en-us_topic_0000002104910598_en-us_topic_0225421598_row1747943121114"><td class="cellrowborder" valign="top" width="32.019999999999996%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000002104910598_p7482142812531"><a name="en-us_topic_0000002104910598_p7482142812531"></a><a name="en-us_topic_0000002104910598_p7482142812531"></a>file:line</p>
</td>
<td class="cellrowborder" valign="top" width="67.97999999999999%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000002104910598_p836421918532"><a name="en-us_topic_0000002104910598_p836421918532"></a><a name="en-us_topic_0000002104910598_p836421918532"></a>File name:Code line number.</p>
</td>
</tr>
<tr id="en-us_topic_0000002104910598_en-us_topic_0225421598_row2706101117135"><td class="cellrowborder" valign="top" width="32.019999999999996%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000002104910598_en-us_topic_0225421598_p3891647181319"><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p3891647181319"></a><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p3891647181319"></a>error code</p>
</td>
<td class="cellrowborder" valign="top" width="67.97999999999999%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000002104910598_p93135259532"><a name="en-us_topic_0000002104910598_p93135259532"></a><a name="en-us_topic_0000002104910598_p93135259532"></a>Error codes of critical and some error logs. For details, see <span id="ph45021084320"><a name="ph45021084320"></a><a name="ph45021084320"></a><em id="en-us_topic_0000002541420481_i69291119573"><a name="en-us_topic_0000002541420481_i69291119573"></a><a name="en-us_topic_0000002541420481_i69291119573"></a><a href="https://www.hiascend.com/document/detail/en/mindie/230/reference/errorcodereference/mindie_log_0072.html" target="_blank" rel="noopener noreferrer">MindIE Error Code Reference</a></em></span>.</p>
</td>
</tr>
<tr id="en-us_topic_0000002104910598_en-us_topic_0225421598_row1570661161316"><td class="cellrowborder" valign="top" width="32.019999999999996%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000002104910598_en-us_topic_0225421598_p690347101310"><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p690347101310"></a><a name="en-us_topic_0000002104910598_en-us_topic_0225421598_p690347101310"></a><strong id="en-us_topic_0000002104910598_b17138157545"><a name="en-us_topic_0000002104910598_b17138157545"></a><a name="en-us_topic_0000002104910598_b17138157545"></a>log message</strong></p>
</td>
<td class="cellrowborder" valign="top" width="67.97999999999999%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000002104910598_p133121025125312"><a name="en-us_topic_0000002104910598_p133121025125312"></a><a name="en-us_topic_0000002104910598_p133121025125312"></a>Error message.</p>
</td>
</tr>
</tbody>
</table>
