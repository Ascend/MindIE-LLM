# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
import json
from typing import Dict, List, Literal, Union, Optional, Tuple, Iterator
from pydantic import BaseModel, model_validator
from ..base.input_builder import InputBuilder
from ...models import TruncationSide


AUTO = 'auto'
ZH = 'zh'
EN = 'en'

DEFAULT_SYSTEM_MESSAGE = 'You are a helpful assistant.'

ROLE = 'role'
CONTENT = 'content'
NAME = 'name'

SYSTEM = 'system'
USER = 'user'
ASSISTANT = 'assistant'
FUNCTION = 'function'
TOOL = 'tool'

FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'

FN_STOP_WORDS = [FN_RESULT, FN_EXIT]

FN_CALL_TEMPLATE_INFO_ZH = """# 工具

## 你拥有如下工具：

{tool_descs}"""

FN_CALL_TEMPLATE_INFO_EN = """# Tools

## You have access to the following tools:

{tool_descs}"""

FN_CALL_TEMPLATE_FMT_ZH = """## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

%s: 工具名称，必须是[{tool_names}]之一。
%s: 工具输入
%s: 工具结果
%s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_EN = """## When you need to call a tool, please insert the following command in your reply, \
which can be called zero or multiple times according to your needs:

%s: The input of the tool
%s: The tool to use, should be one of [{tool_names}]
%s: Tool results
%s: Reply based on tool results. Images need to be rendered as ![](url)""" % (

    FN_ARGS,
    FN_NAME,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_PARA_ZH = """## 你可以在回复中插入以下命令以并行调用N个工具：

%s: 工具1的名称，必须是[{tool_names}]之一
%s: 工具1的输入
%s: 工具2的名称
%s: 工具2的输入
...
%s: 工具N的名称
%s: 工具N的输入
%s: 工具1的结果
%s: 工具2的结果
...
%s: 工具N的结果
%s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_RESULT,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_PARA_EN = """## Insert the following command in your reply when you need \
to call N tools in parallel:

%s: The name of tool 1, should be one of [{tool_names}]
%s: The input of tool 1
%s: The name of tool 2
%s: The input of tool 2
...
%s: The name of tool N
%s: The input of tool N
%s: The result of tool 1
%s: The result of tool 2
...
%s: The result of tool N
%s: Reply based on tool results. Images need to be rendered as ![](url)""" % (
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_RESULT,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE = {
    'zh': FN_CALL_TEMPLATE_INFO_ZH + '\n\n' + FN_CALL_TEMPLATE_FMT_ZH,
    'en': FN_CALL_TEMPLATE_INFO_EN + '\n\n' + FN_CALL_TEMPLATE_FMT_EN,
    'zh_parallel': FN_CALL_TEMPLATE_INFO_ZH + '\n\n' + FN_CALL_TEMPLATE_FMT_PARA_ZH,
    'en_parallel': FN_CALL_TEMPLATE_INFO_EN + '\n\n' + FN_CALL_TEMPLATE_FMT_PARA_EN,
}


class BaseModelCompatibleDict(BaseModel):

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return f'{self.model_dump()}'

    def model_dump(self, **kwargs):
        return super().model_dump(exclude_none=True, **kwargs)

    def model_dump_json(self, **kwargs):
        return super().model_dump_json(exclude_none=True, **kwargs)

    def get(self, key, default=None):
        try:
            value = getattr(self, key)
            if value:
                return value
            else:
                return default
        except AttributeError:
            return default


class Function(BaseModelCompatibleDict):
    name: str
    arguments: str

    def __init__(self, name: str, arguments: str):
        super().__init__(name=name, arguments=arguments)

    def __repr__(self):
        return f'Function({self.model_dump()})'


class ToolCall(BaseModelCompatibleDict):
    function: Function

    def __init__(self, function: Function):
        super().__init__(function=function)

    def __repr__(self):
        return f'ToolCall({self.model_dump()})'


class ContentItem(BaseModelCompatibleDict):
    text: Optional[str] = None
    image: Optional[str] = None
    file: Optional[str] = None

    def __init__(self, text: Optional[str] = None, image: Optional[str] = None, file: Optional[str] = None):
        super().__init__(text=text, image=image, file=file)

    def __repr__(self):
        return f'ContentItem({self.model_dump()})'

    @property
    def type(self) -> Literal['text', 'image', 'file']:
        t, v = self.get_type_and_value()
        return t

    @property
    def value(self) -> str:
        t, v = self.get_type_and_value()
        return v

    @model_validator(mode='after')
    def check_exclusivity(self):
        provided_fields = 0
        if self.text is not None:
            provided_fields += 1
        if self.image:
            provided_fields += 1
        if self.file:
            provided_fields += 1

        if provided_fields != 1:
            raise ValueError("Exactly one of 'text', 'image', or 'file' must be provided.")
        return self

    def get_type_and_value(self) -> Tuple[Literal['text', 'image', 'file'], str]:
        (t, v), = self.model_dump().items()
        return t, v


class Message(BaseModelCompatibleDict):
    role: str
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None
    function_call: Optional[Function] = None
    tool_calls: Optional[List[ToolCall]]
    extra: Optional[dict] = None

    def __init__(self,
                 role: str,
                 content: Optional[Union[str, List[ContentItem]]] = None,
                 name: Optional[str] = None,
                 function_call: Optional[Function] = None,
                 tool_calls: Optional[ToolCall] = None,
                 extra: Optional[dict] = None,
                 **kwargs):
        if content is None:
            content = ''
        super().__init__(role=role, content=content,
                         name=name, function_call=function_call, tool_calls=tool_calls, extra=extra)

    def __repr__(self):
        return f'Message({self.model_dump()})'
    

def get_function_description(function: Dict, lang: Literal[EN, ZH]) -> str:
    """
    Text description of function
    """
    function = function['function']
    tool_desc_template = {
        ZH: '### {name_for_human}\n\n{name_for_model}: {description_for_model} 输入参数：{parameters} {args_format}',
        EN: '### {name_for_human}\n\n{name_for_model}: {description_for_model} Parameters: {parameters} {args_format}'
    }
    tool_desc = tool_desc_template.get(lang)
    name = function.get('name', None)
    name_for_human = function.get('name_for_human', name)
    name_for_model = function.get('name_for_model', name)

    if name_for_model == 'code_interpreter':
        args_format = {
            ZH: '此工具的输入应为Markdown代码块。',
            EN: 'Enclose the code within triple backticks (`) at the beginning and end of the code.',
        }
    else:
        args_format = {
            ZH: '此工具的输入应为JSON对象。',
            EN: 'Format the arguments as a JSON object.',
        }
    args_format = function.get('args_format', args_format.get(lang))

    return tool_desc.format(name_for_human=name_for_human,
                            name_for_model=name_for_model,
                            description_for_model=function['description'],
                            parameters=json.dumps(function['parameters'], ensure_ascii=False),
                            args_format=args_format).rstrip()


def extract_text_from_message(msg: Message,) -> str:
    if isinstance(msg.content, str):
        text = msg.content
    else:
        raise TypeError(f'List of str or str expected, but received {type(msg.content).__name__}.')
    return text.strip()
        



class Qwen2InputBuilder(InputBuilder):

    def __init__(self, tokenizer, is_qwen1_5_or_2, **kwargs):
        self.is_qwen1_5_or_2 = is_qwen1_5_or_2
        self.content_key = "content"
        self.tools_key = "tools"
        self.role_key = "role"
        self.fncall_prompt = None
        super().__init__(tokenizer, **kwargs)

    def simulate_response_completion_with_chat(self, messages: List[Message]) -> List[Message]:
        if messages and (messages[-1].role == ASSISTANT):
            usr = messages[-2].content
            bot = messages[-1].content
            sep = '\n\n'
            if isinstance(usr, str) and isinstance(bot, str):
                usr = usr + sep + bot
            elif isinstance(usr, list) and isinstance(bot, list):
                usr = usr + [ContentItem(text=sep)] + bot
            else:
                raise NotImplementedError
            text_to_complete = copy.deepcopy(messages[-2])
            text_to_complete.content = usr
            messages = messages[:-2] + [text_to_complete]
        res = []
        for message in messages:
            message_dict = {self.role_key: message.role, self.content_key: message.content}
            res.append(message_dict)
        return res

    def _apply_chat_template(self, conversation, tools_msg=None, **kwargs):
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("Your transformers version is detected to be <4.34. This message indicates that this "
                               "model is not supported to run on transformers <4.34. You can upgrade transformers to "
                               "4.34 or above, or rewrite the InputBuilder provided with this model and load it in the "
                               "router.")
        if not self.tokenizer.chat_template:
            raise RuntimeError("The model does not appear to be a chat model because it is not configured with a "
                               "`chat_template`.")
        truncation_method = kwargs.get(self.chat_template_kwargs, {}).get(self.truncation, TruncationSide.RIGHT)
        max_length = kwargs.get(self.chat_template_kwargs, {}).get("max_length", None)
        if truncation_method == TruncationSide.RIGHT:
            kwargs[self.truncation] = True
            kwargs["tokenize"] = True
            kwargs["max_length"] = max_length
        if truncation_method == TruncationSide.LEFT:
            kwargs[self.truncation] = False
            kwargs["tokenize"] = True
        kwargs.pop(self.chat_template_kwargs)
        if self.is_qwen1_5_or_2:  # enter qwen1.5/qwen2 router
            roles = [conv.get('role') for conv in conversation]
            if tools_msg is None and 'tool' not in roles:
                input_ids = self.tokenizer.apply_chat_template(conversation, **kwargs)
            else:
                self.fncall_prompt = QwenFnCallPrompt()
                messages = []
                for msg in conversation:
                    if isinstance(msg, dict):
                        messages.append(Message(**msg))
                    else:
                        messages.append(msg)
                if messages[0]['role'] != SYSTEM:
                    messages = [Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE)] + messages
                if tools_msg is None:
                    functions = None
                    functions_choice = 'none'
                else:
                    functions = tools_msg.get(self.tools_key)
                    functions_choice = tools_msg.get('tool_choice')
                messages = self.fncall_prompt.preprocess_fncall_messages(
                    messages=messages,
                    functions=functions,
                    lang='en',
                    parallel_function_calls=False,
                    function_choice=functions_choice
                )
                messages = self.simulate_response_completion_with_chat(messages)
                input_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
            if truncation_method == TruncationSide.LEFT and len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
            return input_ids
        else:  # enter qwen2.5 router
            if tools_msg:
                tools_list = tools_msg.get("tools", None)
                # tools call need transformers>=4.43.1
                input_ids = self.tokenizer.apply_chat_template(conversation, tools=tools_list, **kwargs)
                if(truncation_method == TruncationSide.LEFT and len(input_ids) > max_length):
                    input_ids = input_ids[-max_length:]
                return input_ids
            input_ids = self.tokenizer.apply_chat_template(conversation, **kwargs)
            if(truncation_method == TruncationSide.LEFT and len(input_ids) > max_length):
                input_ids = input_ids[-max_length:]
            return input_ids


class QwenFnCallPrompt(object):

    @staticmethod
    def preprocess_fncall_messages(
        messages: List[Message],
        functions: List[dict],
        lang: Literal[EN, ZH],
        parallel_function_calls: bool = True,
        function_choice: Union[Literal[AUTO], str] = AUTO,
    ) -> List[Message]:
        ori_messages = messages

        # Change function_call responses to plaintext responses:
        messages = []
        for msg in copy.deepcopy(ori_messages):
            role, content = msg.role, msg.content
            if role in (SYSTEM, USER):
                messages.append(msg)
            elif role == ASSISTANT:
                fn_call = msg.tool_calls[0]
                if fn_call:
                    f_name = fn_call.function.name
                    f_args = fn_call.function.arguments
                    if f_args.startswith('```'):  # if code snippet
                        f_args = '\n' + f_args  # for markdown rendering
                    func_content = '\n' if messages[-1].role == ASSISTANT else ''
                    func_content += f'{FN_NAME}: {f_name}'
                    func_content += f'\n{FN_ARGS}: {f_args}'
                    content += func_content
                if messages[-1].role == ASSISTANT:
                    messages[-1].content += content
                else:
                    messages.append(Message(role=role, content=content))
            elif role == TOOL:
                if content:
                    f_result = copy.deepcopy(content)
                else:
                    f_result = ''
                f_exit = f'\n{FN_EXIT}: '
                last_text_content = messages[-1].content
                if last_text_content.endswith(f_exit):
                    messages[-1].content = last_text_content[:-len(f_exit)]
                f_result = f'\n{FN_RESULT}: ' + f_result + f_exit
                messages[-1].content += f_result
            else:
                raise TypeError

        # Add a system prompt for function calling:
        if functions is not None:
            tool_desc_template = FN_CALL_TEMPLATE[lang + ('_parallel' if parallel_function_calls else '')]
            tool_descs = '\n\n'.join(get_function_description(function, lang=lang) for function in functions)
            tool_names = ','.join(function['function'].get('name_for_model', function['function'].get('name', '')) \
                                  for function in functions)
            tool_system = tool_desc_template.format(tool_descs=tool_descs, tool_names=tool_names)
            if messages[0].role == SYSTEM:
                messages[0].content = messages[0].content + '\n\n' + tool_system
            else:
                messages = [Message(role=SYSTEM, content=[ContentItem(text=tool_system)])] + messages

        # Remove ': ' for continued generation of function calling,
        # because ': ' may form a single token with its following words:
        if messages[-1].role == ASSISTANT:
            last_msg = messages[-1].content
            if last_msg.endswith(f'{FN_EXIT}: '):
                messages[-1].content = messages[-1].content[:-2]

        # Add the function_choice prefix:
        if function_choice not in (AUTO, 'none'):
            if messages[-1].role == ASSISTANT:
                last_msg = messages[-1]
                if last_msg.content:
                    if extract_text_from_message(last_msg).endswith(FN_EXIT):
                        last_msg.content += ': \n'
                    else:
                        last_msg.content += '\n'
                messages = messages[:-1]
            else:
                last_msg = Message(role=ASSISTANT, content='')
            last_msg.content = f'{FN_NAME}: {function_choice}'
            messages = messages + [last_msg]
        return messages


