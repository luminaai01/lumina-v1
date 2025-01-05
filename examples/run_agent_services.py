import asyncio
import time

from lumina.agents.stream import PLUGIN_CN, get_plugin_prompt
from lumina.distributed import AsyncHTTPAgentClient, AsyncHTTPAgentServer, HTTPAgentClient, HTTPAgentServer
from lumina.llms import lumina2_META
from lumina.schema import AgentMessage
from lumina.utils import create_object

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

server = HTTPAgentServer(
    '1',
    {
        'type': 'lumina.agents.AsyncAgent',
        'llm': {
            'type': 'lumina.llms.AsyncLMDeployPipeline',
            'path': 'lumina/lumina2_5-7b-chat',
            'meta_template': lumina2_META,
        }
    },
    port=8090,
)
print(server.is_alive)
message = AgentMessage(sender='user', content='hello')
result = server(message)
print(result)
server.shutdown()

# math coder
server = AsyncHTTPAgentServer(
    '1',
    {
        'type': 'lumina.agents.AsyncMathCoder',
        'llm': {
            'type': 'lumina.llms.AsyncLMDeployPipeline',
            'path': 'lumina/lumina2_5-7b-chat',
            'meta_template': lumina2_META,
            'tp': 1,
            'top_k': 1,
            'temperature': 1.0,
            'stop_words': ['<|im_end|>', '<|action_end|>'],
            'max_new_tokens': 1024,
        },
        'interpreter': {
            'type': 'lumina.actions.AsyncIPythonInterpreter',
            'max_kernels': 100
        },
    },
    port=8091,
)
message = AgentMessage(
    sender='user',
    content=
    ('Marie is thinking of a multiple of 63, while Jay is thinking of a factor '
     'of 63. They happen to be thinking of the same number. There are two '
     'possibilities for the number that each of them is thinking of, one '
     'positive and one negative. Find the product of these two numbers.'))
result = server(message)
print(loop.run_until_complete(result))
print(server.state_dict())

client = AsyncHTTPAgentClient(port=8091)
result = client('hello', session_id=1)
print(loop.run_until_complete(result))
print(client.state_dict(1))

client = HTTPAgentClient(port=8091)
print(client.state_dict(1))
print(client('introduce yourself', session_id=1))
print(client.state_dict(1))
server.shutdown()

# plugins
plugins = [dict(type='lumina.actions.AsyncArxivSearch')]
server_cfg = dict(
    type='lumina.distributed.AsyncHTTPAgentServer',
    gpu_id='1',
    config={
        'type': 'lumina.agents.AsyncAgentForlumina',
        'llm': {
            'type': 'lumina.llms.AsyncLMDeployPipeline',
            'path': 'lumina/lumina2_5-7b-chat',
            'meta_template': lumina2_META,
            'tp': 1,
            'top_k': 1,
            'temperature': 1.0,
            'stop_words': ['<|im_end|>', '<|action_end|>'],
            'max_new_tokens': 1024,
        },
        'plugins': plugins,
        'output_format': {
            'type': 'lumina.prompts.parsers.PluginParser',
            'template': PLUGIN_CN,
            'prompt': get_plugin_prompt(plugins),
        }
    },
    port=8091,
)
server = create_object(server_cfg)
tic = time.time()
coros = [
    server(query, session_id=i)
    for i, query in enumerate(['LLM智能体方向的最新论文有哪些？'] * 50)
]
res = loop.run_until_complete(asyncio.gather(*coros))
print('-' * 120)
print(f'time elapsed: {time.time() - tic}')
server.shutdown()
